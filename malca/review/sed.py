"""SED photometry normalization, conversion, catalog fetchers, and plotting."""

from __future__ import annotations

import io
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from astropy import units as u


SED_TABLE_NAME = "sed_photometry"

SED_COLUMNS = [
    "candidate_id",
    "source",
    "band",
    "mag",
    "mag_err",
    "mag_system",
    "lambda_eff_angstrom",
    "flux_lambda",
    "flux_lambda_err",
    "lambda_l_lambda",
    "lambda_l_lambda_err",
    "flux_nu_jy",
    "flux_nu_jy_err",
    "sep_arcsec",
    "is_synthetic",
    "is_upper_limit",
    "quality_flags",
    "svo_filter_id",
    "av_coeff",
]


@dataclass(frozen=True)
class SedBandpass:
    source: str
    band: str
    mag_col: str | None
    err_col: str | None
    mag_system: str
    lambda_eff_angstrom: float
    fnu_zero_jy: float | None
    av_coeff: float | None = None
    svo_filter_id: str | None = None
    is_synthetic: bool = False
    confusion_risk: bool = False


def _band_key(source: str, band: str) -> str:
    return f"{source.strip().lower()}:{band.strip().lower()}"


def _bp(
    source: str,
    band: str,
    mag_col: str | None,
    err_col: str | None,
    mag_system: str,
    lambda_eff_angstrom: float,
    fnu_zero_jy: float | None = None,
    av_coeff: float | None = None,
    svo_filter_id: str | None = None,
    is_synthetic: bool = False,
    confusion_risk: bool = False,
) -> SedBandpass:
    return SedBandpass(
        source=source,
        band=band,
        mag_col=mag_col,
        err_col=err_col,
        mag_system=mag_system,
        lambda_eff_angstrom=float(lambda_eff_angstrom),
        fnu_zero_jy=fnu_zero_jy,
        av_coeff=av_coeff,
        svo_filter_id=svo_filter_id,
        is_synthetic=is_synthetic,
        confusion_risk=confusion_risk,
    )


# Fallback values are intentionally explicit so SED conversion remains usable
# when the SVO service is unavailable. Zero points are Jy for Vega systems.
SED_BANDPASSES: dict[str, SedBandpass] = {}
for _b in [
    # Existing MALCA payload fields
    _bp("Gaia DR3", "G", "phot_g_mean_mag", None, "Vega", 6730.0, 2861.0, 0.789, "GAIA/GAIA3.G"),
    _bp("Gaia DR3", "BP", "phot_bp_mean_mag", None, "Vega", 5320.0, 3552.0, 1.002, "GAIA/GAIA3.Gbp"),
    _bp("Gaia DR3", "RP", "phot_rp_mean_mag", None, "Vega", 7970.0, 2555.0, 0.589, "GAIA/GAIA3.Grp"),
    _bp("GALEX", "FUV", "galex_fuv", "galex_fuv_err", "AB", 1538.6, None, 2.61, "GALEX/GALEX.FUV"),
    _bp("GALEX", "NUV", "galex_nuv", "galex_nuv_err", "AB", 2315.7, None, 2.76, "GALEX/GALEX.NUV"),
    _bp("APASS", "B", "apass_b", "apass_b_err", "Vega", 4380.0, 4063.0, 1.321, "Generic/Johnson.B"),
    _bp("APASS", "V", "apass_v", "apass_v_err", "Vega", 5450.0, 3636.0, 1.000, "Generic/Johnson.V"),
    _bp("APASS", "g", "apass_g", "apass_g_err", "AB", 4770.0, None, 1.199, "SLOAN/SDSS.g"),
    _bp("APASS", "r", "apass_r", "apass_r_err", "AB", 6231.0, None, 0.858, "SLOAN/SDSS.r"),
    _bp("APASS", "i", "apass_i", "apass_i_err", "AB", 7625.0, None, 0.639, "SLOAN/SDSS.i"),
    _bp("2MASS", "J", "tmass_j", "tmass_j_err", "Vega", 12350.0, 1594.0, 0.282, "2MASS/2MASS.J"),
    _bp("2MASS", "H", "tmass_h", "tmass_h_err", "Vega", 16620.0, 1024.0, 0.175, "2MASS/2MASS.H"),
    _bp("2MASS", "Ks", "tmass_k", "tmass_k_err", "Vega", 21590.0, 666.7, 0.112, "2MASS/2MASS.Ks"),
    _bp("AllWISE", "W1", "w1", "w1_err", "Vega", 33526.0, 309.540, 0.061, "WISE/WISE.W1"),
    _bp("AllWISE", "W2", "w2", "w2_err", "Vega", 46028.0, 171.787, 0.047, "WISE/WISE.W2"),
    _bp("AllWISE", "W3", "w3", "w3_err", "Vega", 115608.0, 31.674, 0.0, "WISE/WISE.W3"),
    _bp("AllWISE", "W4", "w4", "w4_err", "Vega", 220883.0, 8.363, 0.0, "WISE/WISE.W4"),
    _bp("IPHAS", "Halpha", "iphas_ha_mag", None, "Vega", 6568.0, 2950.0, 0.815, "INT/IPHAS.Ha"),
    # Added catalog families
    _bp("Gaia GSPC", "SDSS_u", "gspc_sdss_u", "gspc_sdss_u_err", "AB", 3543.0, None, 1.579, "SLOAN/SDSS.u", True),
    _bp("Gaia GSPC", "SDSS_g", "gspc_sdss_g", "gspc_sdss_g_err", "AB", 4770.0, None, 1.199, "SLOAN/SDSS.g", True),
    _bp("Gaia GSPC", "SDSS_r", "gspc_sdss_r", "gspc_sdss_r_err", "AB", 6231.0, None, 0.858, "SLOAN/SDSS.r", True),
    _bp("Gaia GSPC", "SDSS_i", "gspc_sdss_i", "gspc_sdss_i_err", "AB", 7625.0, None, 0.639, "SLOAN/SDSS.i", True),
    _bp("Gaia GSPC", "SDSS_z", "gspc_sdss_z", "gspc_sdss_z_err", "AB", 9134.0, None, 0.453, "SLOAN/SDSS.z", True),
    _bp("Gaia GSPC", "PS1_g", "gspc_ps1_g", "gspc_ps1_g_err", "AB", 4810.0, None, 1.199, "PAN-STARRS/PS1.g", True),
    _bp("Gaia GSPC", "PS1_r", "gspc_ps1_r", "gspc_ps1_r_err", "AB", 6170.0, None, 0.858, "PAN-STARRS/PS1.r", True),
    _bp("Gaia GSPC", "PS1_i", "gspc_ps1_i", "gspc_ps1_i_err", "AB", 7520.0, None, 0.639, "PAN-STARRS/PS1.i", True),
    _bp("Gaia GSPC", "PS1_z", "gspc_ps1_z", "gspc_ps1_z_err", "AB", 8660.0, None, 0.453, "PAN-STARRS/PS1.z", True),
    _bp("Gaia GSPC", "PS1_y", "gspc_ps1_y", "gspc_ps1_y_err", "AB", 9620.0, None, 0.385, "PAN-STARRS/PS1.y", True),
    _bp("Pan-STARRS", "g", "ps1_g", "ps1_g_err", "AB", 4810.0, None, 1.199, "PAN-STARRS/PS1.g"),
    _bp("Pan-STARRS", "r", "ps1_r", "ps1_r_err", "AB", 6170.0, None, 0.858, "PAN-STARRS/PS1.r"),
    _bp("Pan-STARRS", "i", "ps1_i", "ps1_i_err", "AB", 7520.0, None, 0.639, "PAN-STARRS/PS1.i"),
    _bp("Pan-STARRS", "z", "ps1_z", "ps1_z_err", "AB", 8660.0, None, 0.453, "PAN-STARRS/PS1.z"),
    _bp("Pan-STARRS", "y", "ps1_y", "ps1_y_err", "AB", 9620.0, None, 0.385, "PAN-STARRS/PS1.y"),
    _bp("SDSS", "u", "sdss_u", "sdss_u_err", "AB", 3543.0, None, 1.579, "SLOAN/SDSS.u"),
    _bp("SDSS", "g", "sdss_g", "sdss_g_err", "AB", 4770.0, None, 1.199, "SLOAN/SDSS.g"),
    _bp("SDSS", "r", "sdss_r", "sdss_r_err", "AB", 6231.0, None, 0.858, "SLOAN/SDSS.r"),
    _bp("SDSS", "i", "sdss_i", "sdss_i_err", "AB", 7625.0, None, 0.639, "SLOAN/SDSS.i"),
    _bp("SDSS", "z", "sdss_z", "sdss_z_err", "AB", 9134.0, None, 0.453, "SLOAN/SDSS.z"),
    _bp("SkyMapper", "u", "skymapper_u", "skymapper_u_err", "AB", 3490.0, None, 1.579),
    _bp("SkyMapper", "v", "skymapper_v", "skymapper_v_err", "AB", 3840.0, None, 1.420),
    _bp("SkyMapper", "g", "skymapper_g", "skymapper_g_err", "AB", 5100.0, None, 1.199),
    _bp("SkyMapper", "r", "skymapper_r", "skymapper_r_err", "AB", 6170.0, None, 0.858),
    _bp("SkyMapper", "i", "skymapper_i", "skymapper_i_err", "AB", 7790.0, None, 0.639),
    _bp("SkyMapper", "z", "skymapper_z", "skymapper_z_err", "AB", 9160.0, None, 0.453),
    _bp("DES", "g", "des_g", "des_g_err", "AB", 4770.0, None, 1.199),
    _bp("DES", "r", "des_r", "des_r_err", "AB", 6400.0, None, 0.858),
    _bp("DES", "i", "des_i", "des_i_err", "AB", 7830.0, None, 0.639),
    _bp("DES", "z", "des_z", "des_z_err", "AB", 9170.0, None, 0.453),
    _bp("DES", "Y", "des_y", "des_y_err", "AB", 9890.0, None, 0.385),
    _bp("DECaPS", "g", "decaps_g", "decaps_g_err", "AB", 4770.0, None, 1.199),
    _bp("DECaPS", "r", "decaps_r", "decaps_r_err", "AB", 6400.0, None, 0.858),
    _bp("DECaPS", "i", "decaps_i", "decaps_i_err", "AB", 7830.0, None, 0.639),
    _bp("DECaPS", "z", "decaps_z", "decaps_z_err", "AB", 9170.0, None, 0.453),
    _bp("DECaPS", "Y", "decaps_y", "decaps_y_err", "AB", 9890.0, None, 0.385),
    _bp("UKIDSS", "Y", "ukidss_y", "ukidss_y_err", "Vega", 10300.0, 2026.0, 0.38),
    _bp("UKIDSS", "J", "ukidss_j", "ukidss_j_err", "Vega", 12500.0, 1530.0, 0.282),
    _bp("UKIDSS", "H", "ukidss_h", "ukidss_h_err", "Vega", 16350.0, 1019.0, 0.175),
    _bp("UKIDSS", "K", "ukidss_k", "ukidss_k_err", "Vega", 22000.0, 631.0, 0.112),
    _bp("VISTA/VVV", "Z", "vista_z", "vista_z_err", "Vega", 8780.0, 2217.0, 0.453),
    _bp("VISTA/VVV", "Y", "vista_y", "vista_y_err", "Vega", 10200.0, 2026.0, 0.385),
    _bp("VISTA/VVV", "J", "vista_j", "vista_j_err", "Vega", 12500.0, 1530.0, 0.282),
    _bp("VISTA/VVV", "H", "vista_h", "vista_h_err", "Vega", 16350.0, 1019.0, 0.175),
    _bp("VISTA/VVV", "Ks", "vista_ks", "vista_ks_err", "Vega", 21500.0, 631.0, 0.112),
    _bp("VPHAS+", "u", "vphas_u", "vphas_u_err", "AB", 3543.0, None, 1.579),
    _bp("VPHAS+", "g", "vphas_g", "vphas_g_err", "AB", 4770.0, None, 1.199),
    _bp("VPHAS+", "r", "vphas_r", "vphas_r_err", "AB", 6231.0, None, 0.858),
    _bp("VPHAS+", "i", "vphas_i", "vphas_i_err", "AB", 7625.0, None, 0.639),
    _bp("VPHAS+", "Halpha", "vphas_ha", "vphas_ha_err", "AB", 6568.0, None, 0.815),
    _bp("Spitzer SEIP", "IRAC1", "spitzer_irac1", "spitzer_irac1_err", "Vega", 35500.0, 280.9, 0.0, confusion_risk=True),
    _bp("Spitzer SEIP", "IRAC2", "spitzer_irac2", "spitzer_irac2_err", "Vega", 44930.0, 179.7, 0.0, confusion_risk=True),
    _bp("Spitzer SEIP", "IRAC3", "spitzer_irac3", "spitzer_irac3_err", "Vega", 57310.0, 115.0, 0.0, confusion_risk=True),
    _bp("Spitzer SEIP", "IRAC4", "spitzer_irac4", "spitzer_irac4_err", "Vega", 78720.0, 64.13, 0.0, confusion_risk=True),
    _bp("Spitzer SEIP", "MIPS24", "spitzer_mips24", "spitzer_mips24_err", "Vega", 235000.0, 7.17, 0.0, confusion_risk=True),
    _bp("AKARI", "S9W", "akari_s9w", "akari_s9w_err", "Jy", 90000.0, None, 0.0, confusion_risk=True),
    _bp("AKARI", "L18W", "akari_l18w", "akari_l18w_err", "Jy", 180000.0, None, 0.0, confusion_risk=True),
    _bp("AKARI", "N60", "akari_n60", "akari_n60_err", "Jy", 650000.0, None, 0.0, confusion_risk=True),
    _bp("AKARI", "WIDE-S", "akari_wide_s", "akari_wide_s_err", "Jy", 900000.0, None, 0.0, confusion_risk=True),
    _bp("AKARI", "WIDE-L", "akari_wide_l", "akari_wide_l_err", "Jy", 1400000.0, None, 0.0, confusion_risk=True),
    _bp("AKARI", "N160", "akari_n160", "akari_n160_err", "Jy", 1600000.0, None, 0.0, confusion_risk=True),
    _bp("IRAS", "12", "iras_12", "iras_12_err", "Jy", 120000.0, None, 0.0, confusion_risk=True),
    _bp("IRAS", "25", "iras_25", "iras_25_err", "Jy", 250000.0, None, 0.0, confusion_risk=True),
    _bp("IRAS", "60", "iras_60", "iras_60_err", "Jy", 600000.0, None, 0.0, confusion_risk=True),
    _bp("IRAS", "100", "iras_100", "iras_100_err", "Jy", 1000000.0, None, 0.0, confusion_risk=True),
    _bp("Herschel", "PACS70", "herschel_pacs70", "herschel_pacs70_err", "Jy", 700000.0, None, 0.0, confusion_risk=True),
    _bp("Herschel", "PACS100", "herschel_pacs100", "herschel_pacs100_err", "Jy", 1000000.0, None, 0.0, confusion_risk=True),
    _bp("Herschel", "PACS160", "herschel_pacs160", "herschel_pacs160_err", "Jy", 1600000.0, None, 0.0, confusion_risk=True),
]:
    SED_BANDPASSES[_band_key(_b.source, _b.band)] = _b

PAYLOAD_BANDPASSES = tuple(
    b for b in SED_BANDPASSES.values()
    if b.mag_col
    and (
        b.source
        in {"Gaia DR3", "GALEX", "APASS", "2MASS", "AllWISE", "IPHAS"}
        or b.source.startswith("Gaia GSPC")
    )
)

SOURCE_COLORS = {
    "Gaia DR3": "#d7b43c",
    "Gaia GSPC": "#a78bfa",
    "GALEX": "#48a6ff",
    "APASS": "#41b883",
    "Pan-STARRS": "#2aa198",
    "SDSS": "#5e9cff",
    "SkyMapper": "#00b8a9",
    "DES": "#7fd34e",
    "DECaPS": "#9acd32",
    "2MASS": "#ffb347",
    "UKIDSS": "#ff8c42",
    "VISTA/VVV": "#ff6f61",
    "AllWISE": "#e25555",
    "Spitzer SEIP": "#d95f02",
    "AKARI": "#b2182b",
    "IRAS": "#8b0000",
    "Herschel": "#6a3d9a",
    "IPHAS": "#f781bf",
    "VPHAS+": "#e377c2",
}


def _safe_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        if value is None:
            return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _to_bool_int(value: object) -> int:
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "true", "t", "yes", "y"} else 0
    return 1 if bool(value) else 0


def distance_pc_from_payload(payload: dict) -> float | None:
    distance = _safe_float(payload.get("distance_gspphot"))
    if distance is not None and distance > 0:
        return distance
    parallax = _safe_float(payload.get("parallax"))
    if parallax is not None and parallax > 0:
        return 1000.0 / parallax
    return None


def extinction_av_from_payload(payload: dict, r_v: float = 3.1) -> float | None:
    av = _safe_float(payload.get("A_v_3d"))
    if av is not None and av >= 0:
        return av
    ebv = _safe_float(payload.get("ebv_3d"))
    if ebv is not None and ebv >= 0:
        return float(r_v) * ebv
    return None


def bandpass_for(source: str, band: str) -> SedBandpass | None:
    return SED_BANDPASSES.get(_band_key(source, band))


def flux_nu_jy_from_mag(mag: float, bandpass: SedBandpass) -> float | None:
    system = bandpass.mag_system.strip().upper()
    if system == "JY":
        return float(mag)
    zero_jy = 3631.0 if system == "AB" else bandpass.fnu_zero_jy
    if zero_jy is None or zero_jy <= 0:
        return None
    return float(zero_jy) * 10.0 ** (-0.4 * float(mag))


def mag_from_flux_nu_jy(flux_jy: float) -> float | None:
    flux = _safe_float(flux_jy)
    if flux is None or flux <= 0:
        return None
    return -2.5 * math.log10(flux / 3631.0)


def flux_lambda_from_flux_nu_jy(flux_jy: float, lambda_angstrom: float) -> float | None:
    flux = _safe_float(flux_jy)
    lam = _safe_float(lambda_angstrom)
    if flux is None or flux <= 0 or lam is None or lam <= 0:
        return None
    wavelength = lam * u.AA
    return (flux * u.Jy).to(
        u.erg / u.s / u.cm**2 / u.AA,
        equivalencies=u.spectral_density(wavelength),
    ).value


def lambda_l_lambda_from_flux_lambda(
    flux_lambda: float,
    lambda_angstrom: float,
    distance_pc: float | None,
) -> float | None:
    fl = _safe_float(flux_lambda)
    lam = _safe_float(lambda_angstrom)
    dist = _safe_float(distance_pc)
    if fl is None or fl <= 0 or lam is None or lam <= 0 or dist is None or dist <= 0:
        return None
    distance_cm = (dist * u.pc).to(u.cm).value
    return 4.0 * math.pi * distance_cm**2 * lam * fl


def _row_from_bandpass(
    *,
    candidate_id: str,
    bandpass: SedBandpass,
    mag: float,
    mag_err: float | None,
    distance_pc: float | None,
    av: float | None,
    dereddened: bool,
    sep_arcsec: float | None = None,
    quality_flags: str = "",
    is_upper_limit: bool = False,
) -> dict | None:
    effective_mag = float(mag)
    system = bandpass.mag_system.strip().upper()
    flags = [x for x in str(quality_flags or "").split(";") if x]
    if bandpass.confusion_risk and "confusion_risk" not in flags:
        flags.append("confusion_risk")
    if dereddened and system != "JY":
        if av is not None and bandpass.av_coeff is not None:
            effective_mag = effective_mag - float(av) * float(bandpass.av_coeff)
            flags.append("ism_corrected")
        else:
            flags.append("no_extinction_coeff")
    flux_nu = flux_nu_jy_from_mag(effective_mag, bandpass)
    if flux_nu is None:
        return None
    flux_lambda = flux_lambda_from_flux_nu_jy(flux_nu, bandpass.lambda_eff_angstrom)
    if flux_lambda is None:
        return None
    flux_nu_err = None
    flux_lambda_err = None
    lambda_l_err = None
    merr = _safe_float(mag_err)
    if merr is not None and merr > 0:
        frac = math.log(10.0) * 0.4 * merr
        flux_nu_err = abs(flux_nu) * frac
        flux_lambda_err = abs(flux_lambda) * frac

    lambda_l = lambda_l_lambda_from_flux_lambda(
        flux_lambda,
        bandpass.lambda_eff_angstrom,
        distance_pc,
    )
    if lambda_l is not None and flux_lambda_err is not None and flux_lambda > 0:
        lambda_l_err = abs(lambda_l) * abs(flux_lambda_err / flux_lambda)

    display_mag = mag_from_flux_nu_jy(flux_nu) if system == "JY" else effective_mag
    return {
        "candidate_id": str(candidate_id),
        "source": bandpass.source,
        "band": bandpass.band,
        "mag": display_mag,
        "mag_err": merr,
        "mag_system": "AB" if system == "JY" else bandpass.mag_system,
        "lambda_eff_angstrom": float(bandpass.lambda_eff_angstrom),
        "flux_lambda": float(flux_lambda),
        "flux_lambda_err": flux_lambda_err,
        "lambda_l_lambda": lambda_l,
        "lambda_l_lambda_err": lambda_l_err,
        "flux_nu_jy": float(flux_nu),
        "flux_nu_jy_err": flux_nu_err,
        "sep_arcsec": sep_arcsec,
        "is_synthetic": int(bandpass.is_synthetic),
        "is_upper_limit": int(is_upper_limit),
        "quality_flags": ";".join(sorted(set(flags))),
        "svo_filter_id": bandpass.svo_filter_id,
        "av_coeff": bandpass.av_coeff,
    }


def rows_from_payload(
    payload: dict,
    *,
    candidate_id: str | None = None,
    extinction_mode: str = "observed",
) -> pd.DataFrame:
    """Build normalized SED rows from fields already present in a candidate payload."""
    cid = str(candidate_id or payload.get("candidate_id") or payload.get("asas_sn_id") or "")
    if not cid:
        cid = "unknown"
    mode = str(extinction_mode or "observed").strip().lower()
    dereddened = mode in {"corrected", "ism-corrected", "ism_corrected", "dereddened"}
    av = extinction_av_from_payload(payload)
    distance_pc = distance_pc_from_payload(payload)

    rows: list[dict] = []
    for bandpass in PAYLOAD_BANDPASSES:
        if not bandpass.mag_col:
            continue
        mag = _safe_float(payload.get(bandpass.mag_col))
        if mag is None:
            continue
        mag_err = _safe_float(payload.get(bandpass.err_col)) if bandpass.err_col else None
        row = _row_from_bandpass(
            candidate_id=cid,
            bandpass=bandpass,
            mag=mag,
            mag_err=mag_err,
            distance_pc=distance_pc,
            av=av,
            dereddened=dereddened,
        )
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows, columns=SED_COLUMNS)


def normalize_external_sed_rows(
    rows: pd.DataFrame | Iterable[dict] | None,
    *,
    payload: dict,
    candidate_id: str,
    extinction_mode: str = "observed",
) -> pd.DataFrame:
    """Normalize stored/catalog SED rows into computed plotting columns."""
    if rows is None:
        return pd.DataFrame(columns=SED_COLUMNS)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=SED_COLUMNS)

    mode = str(extinction_mode or "observed").strip().lower()
    dereddened = mode in {"corrected", "ism-corrected", "ism_corrected", "dereddened"}
    av = extinction_av_from_payload(payload)
    distance_pc = distance_pc_from_payload(payload)

    out: list[dict] = []
    for _, item in frame.iterrows():
        source = str(item.get("source") or "").strip()
        band = str(item.get("band") or "").strip()
        bp = bandpass_for(source, band)
        if bp is None:
            lambda_eff = _safe_float(item.get("lambda_eff_angstrom"))
            mag_system = str(item.get("mag_system") or "AB")
            fnu_zero = None if mag_system.upper() == "AB" else _safe_float(item.get("fnu_zero_jy"))
            if lambda_eff is None:
                continue
            bp = SedBandpass(
                source=source or "Catalog",
                band=band or "?",
                mag_col=None,
                err_col=None,
                mag_system=mag_system,
                lambda_eff_angstrom=lambda_eff,
                fnu_zero_jy=fnu_zero,
                av_coeff=_safe_float(item.get("av_coeff")),
                svo_filter_id=str(item.get("svo_filter_id") or "") or None,
                is_synthetic=bool(_to_bool_int(item.get("is_synthetic", False))),
                confusion_risk="confusion_risk" in str(item.get("quality_flags") or ""),
            )

        mag = _safe_float(item.get("mag"))
        flux_nu = _safe_float(item.get("flux_nu_jy"))
        if mag is None and flux_nu is not None:
            mag = mag_from_flux_nu_jy(flux_nu)
        if mag is None:
            continue
        row_mag = flux_nu if bp.mag_system.strip().upper() == "JY" and flux_nu is not None else mag
        row = _row_from_bandpass(
            candidate_id=str(candidate_id),
            bandpass=bp,
            mag=row_mag,
            mag_err=_safe_float(item.get("mag_err")),
            distance_pc=distance_pc,
            av=av,
            dereddened=dereddened,
            sep_arcsec=_safe_float(item.get("sep_arcsec")),
            quality_flags=str(item.get("quality_flags") or ""),
            is_upper_limit=bool(_to_bool_int(item.get("is_upper_limit", False))),
        )
        if row is not None:
            out.append(row)
    return pd.DataFrame(out, columns=SED_COLUMNS)


def build_sed_dataframe(
    payload: dict,
    *,
    candidate_id: str | None = None,
    external_rows: pd.DataFrame | Iterable[dict] | None = None,
    extinction_mode: str = "observed",
) -> pd.DataFrame:
    cid = str(candidate_id or payload.get("candidate_id") or payload.get("asas_sn_id") or "")
    payload_rows = rows_from_payload(payload, candidate_id=cid, extinction_mode=extinction_mode)
    external = normalize_external_sed_rows(
        external_rows,
        payload=payload,
        candidate_id=cid,
        extinction_mode=extinction_mode,
    )
    if payload_rows.empty:
        combined = external
    elif external.empty:
        combined = payload_rows
    else:
        combined = pd.DataFrame(
            [*payload_rows.to_dict("records"), *external.to_dict("records")],
            columns=SED_COLUMNS,
        )
    if combined.empty:
        return pd.DataFrame(columns=SED_COLUMNS)
    combined["_source_rank"] = combined["source"].astype(str).map(lambda s: 1 if s == "Gaia GSPC" else 0)
    combined = combined.sort_values(["lambda_eff_angstrom", "_source_rank", "source", "band"]).drop(columns=["_source_rank"])
    # Avoid exact duplicate rows when payload and persisted rows overlap.
    combined = combined.drop_duplicates(subset=["candidate_id", "source", "band", "quality_flags"], keep="first")
    return combined.reset_index(drop=True)


def _theme(theme: str | None) -> dict[str, str]:
    mode = str(theme or "black").strip().lower()
    if mode == "white":
        return {"paper": "#ffffff", "plot": "#ffffff", "font": "#1c2733", "grid": "rgba(104,128,149,0.18)", "muted": "#5a6b7b"}
    if mode == "gray":
        return {"paper": "#2e3440", "plot": "#2e3440", "font": "#d8dee9", "grid": "rgba(129,161,193,0.15)", "muted": "#aab6c7"}
    return {"paper": "#0d0d0d", "plot": "#0d0d0d", "font": "#dce8f2", "grid": "rgba(96,116,130,0.22)", "muted": "#9fb6cb"}


def build_sed_figure(
    payload: dict,
    *,
    candidate_id: str | None = None,
    external_rows: pd.DataFrame | Iterable[dict] | None = None,
    extinction_mode: str = "observed",
    theme: str | None = None,
) -> tuple[go.Figure, pd.DataFrame, list[str]]:
    """Return a Plotly SED figure, normalized rows, and warning strings."""
    warnings: list[str] = []
    mode = str(extinction_mode or "observed").strip().lower()
    if mode == "both":
        observed = build_sed_dataframe(payload, candidate_id=candidate_id, external_rows=external_rows, extinction_mode="observed")
        corrected = build_sed_dataframe(payload, candidate_id=candidate_id, external_rows=external_rows, extinction_mode="corrected")
        observed["sed_mode"] = "Observed"
        corrected["sed_mode"] = "ISM-corrected"
        if observed.empty and corrected.empty:
            sed_df = pd.DataFrame(columns=SED_COLUMNS + ["sed_mode"])
        else:
            sed_df = pd.DataFrame(
                [*observed.to_dict("records"), *corrected.to_dict("records")],
                columns=SED_COLUMNS + ["sed_mode"],
            )
    else:
        sed_df = build_sed_dataframe(payload, candidate_id=candidate_id, external_rows=external_rows, extinction_mode=mode)
        sed_df["sed_mode"] = "ISM-corrected" if mode in {"corrected", "ism-corrected", "ism_corrected", "dereddened"} else "Observed"

    spec = _theme(theme)
    fig = go.Figure()
    y_col = "lambda_l_lambda"
    if sed_df.empty or sed_df[y_col].isna().all():
        y_col = "flux_lambda"
        if distance_pc_from_payload(payload) is None:
            warnings.append("No distance available; plotting flux density instead of luminosity.")

    if sed_df.empty:
        fig.add_annotation(text="No SED photometry available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    else:
        plot_df = sed_df.copy()
        plot_df["x"] = pd.to_numeric(plot_df["lambda_eff_angstrom"], errors="coerce")
        plot_df["y"] = pd.to_numeric(plot_df[y_col], errors="coerce")
        plot_df = plot_df[np.isfinite(plot_df["x"]) & np.isfinite(plot_df["y"]) & (plot_df["x"] > 0) & (plot_df["y"] > 0)]
        for (mode_name, source), grp in plot_df.groupby(["sed_mode", "source"], dropna=False):
            color = SOURCE_COLORS.get(str(source), "#bbbbbb")
            symbols = []
            for _, row in grp.iterrows():
                flags = str(row.get("quality_flags") or "")
                if _to_bool_int(row.get("is_upper_limit", 0)):
                    symbols.append("triangle-down")
                elif _to_bool_int(row.get("is_synthetic", 0)):
                    symbols.append("circle-open")
                elif "confusion_risk" in flags:
                    symbols.append("diamond-open")
                else:
                    symbols.append("circle")
            opacity = 0.5 if mode_name == "ISM-corrected" and mode == "both" else 0.9
            y_err_col = "lambda_l_lambda_err" if y_col == "lambda_l_lambda" else "flux_lambda_err"
            y_err = pd.to_numeric(grp.get(y_err_col), errors="coerce") if y_err_col in grp.columns else None
            show_y_err = bool(y_err is not None and np.isfinite(y_err).any())
            fig.add_trace(go.Scatter(
                x=grp["x"],
                y=grp["y"],
                mode="markers",
                name=f"{source} ({mode_name})" if mode == "both" else str(source),
                marker=dict(size=8, color=color, symbol=symbols, opacity=opacity, line=dict(width=1, color=spec["font"])),
                error_y=dict(type="data", array=y_err, visible=show_y_err, thickness=0.8),
                customdata=np.column_stack([
                    grp["band"].astype(str),
                    pd.to_numeric(grp["mag"], errors="coerce"),
                    grp["mag_system"].astype(str),
                    grp["quality_flags"].astype(str),
                ]),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "band: %{customdata[0]}<br>"
                    "lambda: %{x:.5g} A<br>"
                    "mag: %{customdata[1]:.4g} %{customdata[2]}<br>"
                    + ("lambda L_lambda: %{y:.4e} erg/s<br>" if y_col == "lambda_l_lambda" else "F_lambda: %{y:.4e}<br>")
                    + "flags: %{customdata[3]}<extra></extra>"
                ),
            ))

    y_title = "lambda L_lambda [erg s^-1]" if y_col == "lambda_l_lambda" else "F_lambda [erg s^-1 cm^-2 A^-1]"
    fig.update_layout(
        title="Spectral Energy Distribution",
        height=320,
        margin=dict(l=58, r=14, t=36, b=48),
        paper_bgcolor=spec["paper"],
        plot_bgcolor=spec["plot"],
        font=dict(color=spec["font"], size=10),
        legend=dict(orientation="h", y=1.08, x=0.0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(title="lambda [Angstrom]", type="log", gridcolor=spec["grid"], zeroline=False)
    fig.update_yaxes(title=y_title, type="log", gridcolor=spec["grid"], zeroline=False)
    return fig, sed_df, warnings


def load_sed_rows(conn: sqlite3.Connection, candidate_id: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(
            f"SELECT {', '.join(SED_COLUMNS)} FROM {SED_TABLE_NAME} WHERE candidate_id = ?",
            conn,
            params=(str(candidate_id),),
        )
    except Exception:
        return pd.DataFrame(columns=SED_COLUMNS)


def upsert_sed_rows(conn: sqlite3.Connection, rows: pd.DataFrame) -> int:
    if rows is None or rows.empty:
        return 0
    frame = rows.copy()
    for col in SED_COLUMNS:
        if col not in frame.columns:
            frame[col] = None
    frame = frame[SED_COLUMNS]
    count = 0
    placeholders = ", ".join(["?"] * len(SED_COLUMNS))
    assignments = ", ".join([f"{col}=excluded.{col}" for col in SED_COLUMNS if col != "candidate_id"])
    sql = (
        f"INSERT INTO {SED_TABLE_NAME} ({', '.join(SED_COLUMNS)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT(candidate_id, source, band) DO UPDATE SET {assignments}"
    )
    for _, row in frame.iterrows():
        values = []
        for col in SED_COLUMNS:
            value = row[col]
            if col in {"is_synthetic", "is_upper_limit"}:
                values.append(_to_bool_int(value))
            elif value is None:
                values.append(None)
            else:
                try:
                    if pd.isna(value):
                        values.append(None)
                    else:
                        values.append(value)
                except Exception:
                    values.append(value)
        conn.execute(sql, values)
        count += 1
    conn.commit()
    return count


def _candidate_id_for_row(row: pd.Series) -> str:
    for col in ("candidate_id", "asas_sn_id", "gaia_id", "source_id"):
        if col in row and str(row.get(col) or "").strip():
            return str(row.get(col)).strip()
    return str(row.name)


def rows_from_candidate_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        payload = row.to_dict()
        rows.append(rows_from_payload(payload, candidate_id=_candidate_id_for_row(row), extinction_mode="observed"))
    if not rows:
        return pd.DataFrame(columns=SED_COLUMNS)
    return pd.concat(rows, ignore_index=True) if any(not part.empty for part in rows) else pd.DataFrame(columns=SED_COLUMNS)


def _try_requests_get_csv(url: str, timeout: int = 30) -> pd.DataFrame:
    import requests

    res = requests.get(url, timeout=timeout)
    res.raise_for_status()
    if not res.text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(res.text))


def _rows_from_simple_mag_dict(
    candidate_id: str,
    values: dict[str, tuple[float | None, float | None]],
    *,
    source: str,
    distance_pc: float | None,
    sep_arcsec: float | None = None,
    quality_flags: str = "",
) -> list[dict]:
    out = []
    for band, (mag, mag_err) in values.items():
        bp = bandpass_for(source, band)
        if bp is None or mag is None:
            continue
        row = _row_from_bandpass(
            candidate_id=candidate_id,
            bandpass=bp,
            mag=float(mag),
            mag_err=mag_err,
            distance_pc=distance_pc,
            av=None,
            dereddened=False,
            sep_arcsec=sep_arcsec,
            quality_flags=quality_flags,
        )
        if row is not None:
            out.append(row)
    return out


def _row_value(row: pd.Series, aliases: str | Iterable[str] | None) -> object:
    if aliases is None:
        return None
    names = [aliases] if isinstance(aliases, str) else list(aliases)
    for name in names:
        if name in row:
            return row.get(name)
    lookup = {str(c).lower(): c for c in row.index}
    compact = {str(c).lower().replace("_", "").replace("-", ""): c for c in row.index}
    for name in names:
        key = str(name).lower()
        actual = lookup.get(key)
        if actual is not None:
            return row.get(actual)
        actual = compact.get(key.replace("_", "").replace("-", ""))
        if actual is not None:
            return row.get(actual)
    return None


def _ra_dec_from_row(row: pd.Series) -> tuple[float | None, float | None]:
    ra = _safe_float(_row_value(row, ("ra", "ra_deg", "RA", "RAJ2000", "RA_ICRS", "RAICRS")))
    dec = _safe_float(_row_value(row, ("dec", "dec_deg", "DEC", "DEJ2000", "DE_ICRS", "DEICRS")))
    return ra, dec


def query_ps1_mean_photometry(df: pd.DataFrame, radius_arcsec: float = 1.5) -> pd.DataFrame:
    """Fetch PS1 DR2 mean photometry from MAST, best effort."""
    rows: list[dict] = []
    for _, item in df.iterrows():
        ra, dec = _ra_dec_from_row(item)
        if ra is None or dec is None or dec < -30.5:
            continue
        cid = _candidate_id_for_row(item)
        payload = item.to_dict()
        radius_deg = float(radius_arcsec) / 3600.0
        try:
            url = (
                "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean.csv"
                f"?ra={ra}&dec={dec}&radius={radius_deg}&pagesize=20&format=csv"
            )
            result = _try_requests_get_csv(url)
            if result.empty:
                continue
            if "distance" in result.columns:
                result = result.sort_values("distance")
            row = result.iloc[0]
            values = {
                "g": (_safe_float(row.get("gMeanPSFMag")), _safe_float(row.get("gMeanPSFMagErr"))),
                "r": (_safe_float(row.get("rMeanPSFMag")), _safe_float(row.get("rMeanPSFMagErr"))),
                "i": (_safe_float(row.get("iMeanPSFMag")), _safe_float(row.get("iMeanPSFMagErr"))),
                "z": (_safe_float(row.get("zMeanPSFMag")), _safe_float(row.get("zMeanPSFMagErr"))),
                "y": (_safe_float(row.get("yMeanPSFMag")), _safe_float(row.get("yMeanPSFMagErr"))),
            }
            sep = _safe_float(row.get("distance"))
            rows.extend(_rows_from_simple_mag_dict(cid, values, source="Pan-STARRS", distance_pc=distance_pc_from_payload(payload), sep_arcsec=sep))
        except Exception:
            continue
    return pd.DataFrame(rows, columns=SED_COLUMNS)


def query_gaia_gspc_photometry(df: pd.DataFrame, chunk_size: int = 100) -> pd.DataFrame:
    """Fetch Gaia DR3 GSPC synthetic SDSS/PS1 photometry, best effort."""
    ids = []
    id_to_payload: dict[str, dict] = {}
    for _, item in df.iterrows():
        sid = str(item.get("gaia_id") or item.get("source_id") or "").strip()
        if sid and sid.lower() not in {"nan", "<na>"}:
            ids.append(sid)
            id_to_payload[sid] = item.to_dict()
    if not ids:
        return pd.DataFrame(columns=SED_COLUMNS)
    rows: list[dict] = []
    try:
        import pyvo

        tap = pyvo.dal.TAPService("https://gaia.aip.de/tap")
        for start in range(0, len(ids), chunk_size):
            chunk = ids[start:start + chunk_size]
            id_list = ",".join(chunk)
            query = (
                "SELECT source_id, "
                "u_sdss_mag, u_sdss_mag_error, g_sdss_mag, g_sdss_mag_error, "
                "r_sdss_mag, r_sdss_mag_error, i_sdss_mag, i_sdss_mag_error, z_sdss_mag, z_sdss_mag_error, "
                "g_ps1_mag, g_ps1_mag_error, r_ps1_mag, r_ps1_mag_error, i_ps1_mag, i_ps1_mag_error, "
                "z_ps1_mag, z_ps1_mag_error, y_ps1_mag, y_ps1_mag_error "
                "FROM gaiadr3.synthetic_photometry_gspc "
                f"WHERE source_id IN ({id_list})"
            )
            table = tap.search(query).to_table().to_pandas()
            for _, row in table.iterrows():
                sid = str(row.get("source_id")).strip()
                payload = id_to_payload.get(sid, {})
                cid = str(payload.get("candidate_id") or payload.get("asas_sn_id") or sid)
                values = {
                    "SDSS_u": (_safe_float(row.get("u_sdss_mag")), _safe_float(row.get("u_sdss_mag_error"))),
                    "SDSS_g": (_safe_float(row.get("g_sdss_mag")), _safe_float(row.get("g_sdss_mag_error"))),
                    "SDSS_r": (_safe_float(row.get("r_sdss_mag")), _safe_float(row.get("r_sdss_mag_error"))),
                    "SDSS_i": (_safe_float(row.get("i_sdss_mag")), _safe_float(row.get("i_sdss_mag_error"))),
                    "SDSS_z": (_safe_float(row.get("z_sdss_mag")), _safe_float(row.get("z_sdss_mag_error"))),
                    "PS1_g": (_safe_float(row.get("g_ps1_mag")), _safe_float(row.get("g_ps1_mag_error"))),
                    "PS1_r": (_safe_float(row.get("r_ps1_mag")), _safe_float(row.get("r_ps1_mag_error"))),
                    "PS1_i": (_safe_float(row.get("i_ps1_mag")), _safe_float(row.get("i_ps1_mag_error"))),
                    "PS1_z": (_safe_float(row.get("z_ps1_mag")), _safe_float(row.get("z_ps1_mag_error"))),
                    "PS1_y": (_safe_float(row.get("y_ps1_mag")), _safe_float(row.get("y_ps1_mag_error"))),
                }
                rows.extend(_rows_from_simple_mag_dict(cid, values, source="Gaia GSPC", distance_pc=distance_pc_from_payload(payload), quality_flags="synthetic_from_gaia_xp"))
    except Exception:
        return pd.DataFrame(columns=SED_COLUMNS)
    return pd.DataFrame(rows, columns=SED_COLUMNS)


@dataclass(frozen=True)
class VizierSourceSpec:
    source: str
    catalog: str
    radius_arcsec: float
    ra_col: str
    dec_col: str
    bands: dict[str, tuple[str | tuple[str, ...], str | tuple[str, ...] | None]]


VIZIER_SOURCE_SPECS: dict[str, VizierSourceSpec] = {
    "sdss": VizierSourceSpec("SDSS", "V/154/sdss16", 1.5, "RA_ICRS", "DE_ICRS", {
        "u": ("umag", "e_umag"), "g": ("gmag", "e_gmag"), "r": ("rmag", "e_rmag"), "i": ("imag", "e_imag"), "z": ("zmag", "e_zmag"),
    }),
    "skymapper": VizierSourceSpec("SkyMapper", "II/379/smssdr4", 1.5, "RAICRS", "DEICRS", {
        "u": ("uPSF", "e_uPSF"), "v": ("vPSF", "e_vPSF"), "g": ("gPSF", "e_gPSF"), "r": ("rPSF", "e_rPSF"), "i": ("iPSF", "e_iPSF"), "z": ("zPSF", "e_zPSF"),
    }),
    "decaps": VizierSourceSpec("DECaPS", "II/376/decaps2", 1.2, "RA_ICRS", "DE_ICRS", {
        "g": ("gmag", "e_gmag"), "r": ("rmag", "e_rmag"), "i": ("imag", "e_imag"), "z": ("zmag", "e_zmag"), "Y": ("Ymag", "e_Ymag"),
    }),
    "des": VizierSourceSpec("DES", "II/371/des_dr2", 1.2, "RA_ICRS", "DE_ICRS", {
        "g": (("WAVG_MAG_PSF_G", "MAG_PSF_G", "gmag"), ("WAVG_MAGERR_PSF_G", "MAGERR_PSF_G", "e_gmag")),
        "r": (("WAVG_MAG_PSF_R", "MAG_PSF_R", "rmag"), ("WAVG_MAGERR_PSF_R", "MAGERR_PSF_R", "e_rmag")),
        "i": (("WAVG_MAG_PSF_I", "MAG_PSF_I", "imag"), ("WAVG_MAGERR_PSF_I", "MAGERR_PSF_I", "e_imag")),
        "z": (("WAVG_MAG_PSF_Z", "MAG_PSF_Z", "zmag"), ("WAVG_MAGERR_PSF_Z", "MAGERR_PSF_Z", "e_zmag")),
        "Y": (("WAVG_MAG_PSF_Y", "MAG_PSF_Y", "Ymag"), ("WAVG_MAGERR_PSF_Y", "MAGERR_PSF_Y", "e_Ymag")),
    }),
    "ukidss": VizierSourceSpec("UKIDSS", "II/319/las9", 1.2, "RAJ2000", "DEJ2000", {
        "Y": ("Ymag", "e_Ymag"), "J": ("Jmag", "e_Jmag"), "H": ("Hmag", "e_Hmag"), "K": ("Kmag", "e_Kmag"),
    }),
    "vista": VizierSourceSpec("VISTA/VVV", "II/348/vvv2", 1.2, "RAJ2000", "DEJ2000", {
        "Z": ("Zmag", "e_Zmag"), "Y": ("Ymag", "e_Ymag"), "J": ("Jmag", "e_Jmag"), "H": ("Hmag", "e_Hmag"), "Ks": ("Ksmag", "e_Ksmag"),
    }),
    "vphas": VizierSourceSpec("VPHAS+", "II/383/vphas2", 1.0, "RAJ2000", "DEJ2000", {
        "u": ("umag", "e_umag"), "g": ("gmag", "e_gmag"), "r": ("rmag", "e_rmag"), "i": ("imag", "e_imag"), "Halpha": ("Ham", "e_Ham"),
    }),
    "spitzer": VizierSourceSpec("Spitzer SEIP", "II/368/sstsl2", 2.0, "RAJ2000", "DEJ2000", {
        "IRAC1": ("I1mag", "e_I1mag"), "IRAC2": ("I2mag", "e_I2mag"), "IRAC3": ("I3mag", "e_I3mag"), "IRAC4": ("I4mag", "e_I4mag"), "MIPS24": ("M24mag", "e_M24mag"),
    }),
}


def query_vizier_source(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Fetch one VizieR-backed source spec by nearest cone match, best effort."""
    spec = VIZIER_SOURCE_SPECS[key]
    rows: list[dict] = []
    try:
        from astropy.coordinates import SkyCoord
        from astroquery.vizier import Vizier
    except Exception:
        return pd.DataFrame(columns=SED_COLUMNS)

    viz = Vizier(columns=["**"], row_limit=5)
    for _, item in df.iterrows():
        ra, dec = _ra_dec_from_row(item)
        if ra is None or dec is None:
            continue
        cid = _candidate_id_for_row(item)
        payload = item.to_dict()
        try:
            target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            tables = viz.query_region(
                target,
                radius=spec.radius_arcsec * u.arcsec,
                catalog=spec.catalog,
            )
            if not tables:
                continue
            result = tables[0].to_pandas()
            if result.empty:
                continue
            row = result.iloc[0]
            values = {
                band: (_safe_float(_row_value(row, mag_col)), _safe_float(_row_value(row, err_col)) if err_col else None)
                for band, (mag_col, err_col) in spec.bands.items()
            }
            sep_arcsec = None
            match_ra = _safe_float(_row_value(row, (spec.ra_col, "RA_ICRS", "RAJ2000", "RAICRS")))
            match_dec = _safe_float(_row_value(row, (spec.dec_col, "DE_ICRS", "DEJ2000", "DEICRS")))
            if match_ra is not None and match_dec is not None:
                sep_arcsec = float(target.separation(SkyCoord(ra=match_ra * u.deg, dec=match_dec * u.deg)).arcsec)
            rows.extend(_rows_from_simple_mag_dict(
                cid,
                values,
                source=spec.source,
                distance_pc=distance_pc_from_payload(payload),
                sep_arcsec=sep_arcsec,
            ))
            time.sleep(0.02)
        except Exception:
            continue
    return pd.DataFrame(rows, columns=SED_COLUMNS)


def query_des_photometry(df: pd.DataFrame) -> pd.DataFrame:
    return query_vizier_source(df, "des")


@dataclass(frozen=True)
class VizierFluxSpec:
    source: str
    catalog: str
    radius_arcsec: float
    bands: dict[str, tuple[str | tuple[str, ...], str | tuple[str, ...] | None]]


VIZIER_FLUX_SPECS: dict[str, VizierFluxSpec] = {
    "akari": VizierFluxSpec("AKARI", "II/297/irc", 5.0, {
        "S9W": (("S9W", "F09", "Flux9"), ("e_S9W", "e_F09", "e_Flux9")),
        "L18W": (("L18W", "F18", "Flux18"), ("e_L18W", "e_F18", "e_Flux18")),
    }),
    "akari_fis": VizierFluxSpec("AKARI", "II/298/fis", 20.0, {
        "N60": (("F65", "N60", "Flux65"), ("e_F65", "e_N60")),
        "WIDE-S": (("F90", "WIDES", "Flux90"), ("e_F90", "e_WIDES")),
        "WIDE-L": (("F140", "WIDEL", "Flux140"), ("e_F140", "e_WIDEL")),
        "N160": (("F160", "N160", "Flux160"), ("e_F160", "e_N160")),
    }),
    "iras": VizierFluxSpec("IRAS", "II/125/main", 30.0, {
        "12": (("F12", "f12"), ("e_F12",)),
        "25": (("F25", "f25"), ("e_F25",)),
        "60": (("F60", "f60"), ("e_F60",)),
        "100": (("F100", "f100"), ("e_F100",)),
    }),
    "herschel70": VizierFluxSpec("Herschel", "VIII/106/hppsc070", 8.0, {
        "PACS70": (("Flux", "flux", "F70", "Fnu"), ("e_Flux", "e_flux", "e_F70")),
    }),
    "herschel100": VizierFluxSpec("Herschel", "VIII/106/hppsc100", 10.0, {
        "PACS100": (("Flux", "flux", "F100", "Fnu"), ("e_Flux", "e_flux", "e_F100")),
    }),
    "herschel160": VizierFluxSpec("Herschel", "VIII/106/hppsc160", 14.0, {
        "PACS160": (("Flux", "flux", "F160", "Fnu"), ("e_Flux", "e_flux", "e_F160")),
    }),
}


def query_flux_catalog_source(df: pd.DataFrame, source_key: str) -> pd.DataFrame:
    """Fetch a flux-density catalog source from VizieR, best effort."""
    requested_keys = {
        "akari": ("akari", "akari_fis"),
        "iras": ("iras",),
        "herschel": ("herschel70", "herschel100", "herschel160"),
    }.get(source_key, (source_key,))
    try:
        from astropy.coordinates import SkyCoord
        from astroquery.vizier import Vizier
    except Exception:
        return pd.DataFrame(columns=SED_COLUMNS)

    out: list[dict] = []
    viz = Vizier(columns=["**"], row_limit=5)
    for _, item in df.iterrows():
        ra, dec = _ra_dec_from_row(item)
        if ra is None or dec is None:
            continue
        cid = _candidate_id_for_row(item)
        payload = item.to_dict()
        for key in requested_keys:
            spec = VIZIER_FLUX_SPECS.get(key)
            if spec is None:
                continue
            try:
                tables = viz.query_region(
                    SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
                    radius=spec.radius_arcsec * u.arcsec,
                    catalog=spec.catalog,
                )
                if not tables:
                    continue
                result = tables[0].to_pandas()
                if result.empty:
                    continue
                row = result.iloc[0]
                for band, (flux_aliases, err_aliases) in spec.bands.items():
                    flux = _safe_float(_row_value(row, flux_aliases))
                    if flux is None or flux <= 0:
                        continue
                    bp = bandpass_for(spec.source, band)
                    if bp is None:
                        continue
                    flux_err = _safe_float(_row_value(row, err_aliases))
                    sed_row = _row_from_bandpass(
                        candidate_id=cid,
                        bandpass=bp,
                        mag=flux,
                        mag_err=(flux_err / flux / (0.4 * math.log(10.0)) if flux_err is not None and flux_err > 0 else None),
                        distance_pc=distance_pc_from_payload(payload),
                        av=None,
                        dereddened=False,
                        sep_arcsec=None,
                        quality_flags="confusion_risk;flux_catalog",
                    )
                    if sed_row is not None:
                        sed_row["flux_nu_jy"] = flux
                        sed_row["flux_nu_jy_err"] = flux_err
                        sed_row["mag"] = mag_from_flux_nu_jy(flux)
                        sed_row["mag_system"] = "AB"
                        out.append(sed_row)
                time.sleep(0.02)
            except Exception:
                continue
    return pd.DataFrame(out, columns=SED_COLUMNS)


CATALOG_FETCHERS = {
    "payload": rows_from_candidate_frame,
    "gaia_gspc": query_gaia_gspc_photometry,
    "ps1": query_ps1_mean_photometry,
    "sdss": lambda df: query_vizier_source(df, "sdss"),
    "skymapper": lambda df: query_vizier_source(df, "skymapper"),
    "des": query_des_photometry,
    "decaps": lambda df: query_vizier_source(df, "decaps"),
    "ukidss": lambda df: query_vizier_source(df, "ukidss"),
    "vista": lambda df: query_vizier_source(df, "vista"),
    "vphas": lambda df: query_vizier_source(df, "vphas"),
    "spitzer": lambda df: query_vizier_source(df, "spitzer"),
    "akari": lambda df: query_flux_catalog_source(df, "akari"),
    "iras": lambda df: query_flux_catalog_source(df, "iras"),
    "herschel": lambda df: query_flux_catalog_source(df, "herschel"),
}

ALL_CATALOG_SOURCES = tuple(CATALOG_FETCHERS)
FAR_IR_CATALOG_SOURCES = ("akari", "iras", "herschel")
DEFAULT_PIPELINE_SED_SOURCES = tuple(
    source for source in ALL_CATALOG_SOURCES
    if source not in FAR_IR_CATALOG_SOURCES
)


def resolve_sed_sources(sources: Iterable[str] | str = "default") -> tuple[str, ...]:
    if isinstance(sources, str):
        text = sources.strip().lower()
        if text in {"", "default", "pipeline"}:
            return DEFAULT_PIPELINE_SED_SOURCES
        if text == "all":
            return ALL_CATALOG_SOURCES
        if text in {"far_ir", "far-ir", "farir"}:
            return FAR_IR_CATALOG_SOURCES
        requested = tuple(x.strip().lower() for x in text.split(",") if x.strip())
    else:
        requested = tuple(str(x).strip().lower() for x in sources if str(x).strip())
    return requested


def fetch_sed_photometry(df: pd.DataFrame, sources: Iterable[str] | str = "default") -> pd.DataFrame:
    requested = resolve_sed_sources(sources)
    frames = []
    for key in requested:
        fetcher = CATALOG_FETCHERS.get(key)
        if fetcher is None:
            continue
        try:
            part = fetcher(df)
        except Exception:
            part = pd.DataFrame(columns=SED_COLUMNS)
        if part is not None and not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame(columns=SED_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["candidate_id", "source", "band"], keep="first").reset_index(drop=True)
