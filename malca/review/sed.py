"""SED photometry normalization, conversion, catalog fetchers, and plotting."""

from __future__ import annotations

import io
import hashlib
import json
import math
import os
import re
import sqlite3
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from threading import RLock
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from astropy import units as u

from malca.config import DEFAULT_CACHE_DIR, PARQUET_CACHE_COMPRESSION
from malca.extinction import mid_ir_av_coefficient
from malca.plotting.lightcurve_publication import PUBLICATION_PLOTLY_FONT
from malca.review.sed_storage import (
    CANONICAL_SED_NORMALIZATION_VERSION,
    LEGACY_CANONICAL_SED_NORMALIZATION_VERSION,
)

SED_TABLE_NAME = "sed_photometry"
VIZIER_QUERY_TIMEOUT_SEC = 30
SED_CACHE_DIR = DEFAULT_CACHE_DIR.expanduser() / "sed"
SED_CACHE_META_COLUMNS = {
    "_cache_candidate_id",
    "_cache_status",
    "_cache_updated_at",
    "_cache_catalog_release",
    "_cache_adapter_version",
    "_cache_match_policy_version",
    "_cache_coordinate_epoch",
    "_cache_quality_policy_version",
    "_cache_fetch_signature",
    "_cache_astrometry_hash",
}
SED_CACHE_SKIP_SOURCES = {"payload"}
SED_FETCH_MANIFEST_ATTR = "sed_fetch_manifest"
SED_FETCH_POLICY_VERSION = "sed-fetch-v3-archive-ledger"
SED_FETCH_CHUNK_SIZE = max(int(os.environ.get("MALCA_SED_FETCH_CHUNK_SIZE", "500")), 1)
SED_FETCH_MAX_ATTEMPTS = max(int(os.environ.get("MALCA_SED_FETCH_MAX_ATTEMPTS", "3")), 1)
SED_FETCH_RETRY_BASE_SECONDS = max(
    float(os.environ.get("MALCA_SED_FETCH_RETRY_BASE_SECONDS", "1.0")),
    0.0,
)
SED_REQUEST_INTERVAL_SECONDS = max(
    float(os.environ.get("MALCA_SED_REQUEST_INTERVAL_SECONDS", "0.05")),
    0.0,
)
SED_BULK_XMATCH_MIN_CANDIDATES = max(
    int(os.environ.get("MALCA_SED_BULK_XMATCH_MIN_CANDIDATES", "20")),
    2,
)
GAIA_XP_FETCH_CHUNK_SIZE = max(
    int(os.environ.get("MALCA_GAIA_XP_FETCH_CHUNK_SIZE", "20")),
    1,
)
SED_MATCH_AMBIGUITY_GAP_ARCSEC = 0.5
SED_MATCH_EDGE_FRACTION = 0.8
SED_MATCH_HIGH_PM_MASYR = 100.0
LSUN_ERG_S = 3.828e33

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

# Versioned-storage consumers can opt into explicit measurement/calibration
# semantics without changing the legacy ``sed_photometry`` table contract.
# The historical ``lambda_eff_angstrom`` column remains a compatibility alias for
# ``plot_lambda_angstrom``.
SED_SEMANTIC_COLUMNS = [
    "measurement_id",
    "normalization_version",
    "catalog_release",
    "source_object_id",
    "catalog_measurement_id",
    "instrument",
    "exposure_id",
    "epoch_mjd",
    "correlation_group",
    "provenance_json",
    "native_value",
    "native_error",
    "native_unit",
    "observable_kind",
    "passband_fidelity",
    "observed_flux_nu_jy",
    "observed_flux_nu_jy_err",
    "plot_lambda_angstrom",
    "plot_lambda_kind",
    "lambda_nominal_angstrom",
    "lambda_pivot_angstrom",
    "lambda_reference_angstrom",
    "lambda_isophotal_angstrom",
    "response_kind",
    "fit_policy",
    "systematic_floor_mag",
    "native_flux_unit",
    "calibration_source",
    "calibration_id",
    "calibration_hash",
    "response_hash",
    "normalization_hash",
]
CANONICAL_SED_COLUMNS = [*SED_COLUMNS, *SED_SEMANTIC_COLUMNS]

SED_FETCH_STATUS_ATTR = "sed_fetch_status_by_candidate"
SED_CACHE_TERMINAL_STATUSES = frozenset(
    {
        "hit",
        "miss",
        "outside_footprint",
        "not_observed",
        "catalog_no_match",
        "covered_no_detection",
        "catalog_detection",
        "image_detection",
        "upper_limit",
        "ambiguous_counterpart",
        "unusable_measurement",
        "reduction_required",
    }
)
SED_CACHE_RETRYABLE_STATUSES = frozenset({"error", "query_error", "partial"})
SED_MANIFEST_TERMINAL_STATUS_PRIORITY = (
    "image_detection",
    "catalog_detection",
    "hit",
    "upper_limit",
    "ambiguous_counterpart",
    "unusable_measurement",
    "reduction_required",
    "covered_no_detection",
    "catalog_no_match",
    "not_observed",
    "outside_footprint",
    "miss",
)
APASS_B_RED_LEAK_COLOR_THRESHOLD = 3.5


@dataclass(frozen=True)
class SedFetchSignature:
    catalog_release: str
    adapter_version: str
    match_policy_version: str
    coordinate_epoch: str
    quality_policy_version: str


SED_SOURCE_FETCH_SIGNATURES: dict[str, SedFetchSignature] = {}


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
    plot_lambda_kind: str = "legacy_catalog_value"
    lambda_nominal_angstrom: float | None = None
    lambda_pivot_angstrom: float | None = None
    lambda_reference_angstrom: float | None = None
    lambda_isophotal_angstrom: float | None = None
    response_kind: str = "instrument_or_standard_response"
    fit_policy: str = "photosphere_or_diagnostic_by_wavelength"
    systematic_floor_mag: float | None = None
    policy_flags: tuple[str, ...] = ()


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
    *,
    plot_lambda_kind: str | None = None,
    lambda_nominal_angstrom: float | None = None,
    lambda_pivot_angstrom: float | None = None,
    lambda_reference_angstrom: float | None = None,
    lambda_isophotal_angstrom: float | None = None,
    response_kind: str | None = None,
    fit_policy: str = "photosphere_or_diagnostic_by_wavelength",
    systematic_floor_mag: float | None = None,
    policy_flags: Iterable[str] = (),
) -> SedBandpass:
    system = str(mag_system or "").strip().upper()
    if plot_lambda_kind is None:
        plot_lambda_kind = "mission_reference" if system == "JY" else "legacy_catalog_value"
    if system == "JY":
        lambda_reference_angstrom = lambda_reference_angstrom or float(lambda_eff_angstrom)
        response_kind = response_kind or "mission_calibrated_monochromatic_flux"
        fit_policy = "diagnostic_only"
    else:
        lambda_nominal_angstrom = lambda_nominal_angstrom or float(lambda_eff_angstrom)
        response_kind = response_kind or "instrument_or_standard_response"
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
        plot_lambda_kind=str(plot_lambda_kind),
        lambda_nominal_angstrom=lambda_nominal_angstrom,
        lambda_pivot_angstrom=lambda_pivot_angstrom,
        lambda_reference_angstrom=lambda_reference_angstrom,
        lambda_isophotal_angstrom=lambda_isophotal_angstrom,
        response_kind=str(response_kind),
        fit_policy=str(fit_policy),
        systematic_floor_mag=systematic_floor_mag,
        policy_flags=tuple(str(flag) for flag in policy_flags if str(flag)),
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
    _bp(
        "APASS", "B", "apass_b", "apass_b_err", "Vega", 4380.0, 4063.0, 1.321, "Generic/Johnson.B",
        response_kind="standardized_system_proxy",
        fit_policy="photosphere_proxy",
        policy_flags=("standardized_system_proxy",),
    ),
    _bp(
        "APASS", "V", "apass_v", "apass_v_err", "Vega", 5450.0, 3636.0, 1.000, "Generic/Johnson.V",
        response_kind="standardized_system_proxy",
        fit_policy="photosphere_proxy",
        policy_flags=("standardized_system_proxy",),
    ),
    _bp(
        "APASS", "g", "apass_g", "apass_g_err", "AB", 4770.0, None, 1.199, "SLOAN/SDSS.g",
        response_kind="standardized_system_proxy",
        fit_policy="photosphere_proxy",
        policy_flags=("standardized_system_proxy",),
    ),
    _bp(
        "APASS", "r", "apass_r", "apass_r_err", "AB", 6231.0, None, 0.858, "SLOAN/SDSS.r",
        response_kind="standardized_system_proxy",
        fit_policy="photosphere_proxy",
        policy_flags=("standardized_system_proxy",),
    ),
    _bp(
        "APASS", "i", "apass_i", "apass_i_err", "AB", 7625.0, None, 0.639, "SLOAN/SDSS.i",
        response_kind="standardized_system_proxy",
        fit_policy="photosphere_proxy",
        policy_flags=("standardized_system_proxy",),
    ),
    _bp("2MASS", "J", "tmass_j", "tmass_j_err", "Vega", 12350.0, 1594.0, 0.282, "2MASS/2MASS.J"),
    _bp("2MASS", "H", "tmass_h", "tmass_h_err", "Vega", 16620.0, 1024.0, 0.175, "2MASS/2MASS.H"),
    _bp("2MASS", "Ks", "tmass_k", "tmass_k_err", "Vega", 21590.0, 666.7, 0.112, "2MASS/2MASS.Ks"),
    _bp("AllWISE", "W1", "w1", "w1_err", "Vega", 33526.0, 309.540, 0.061, "WISE/WISE.W1"),
    _bp("AllWISE", "W2", "w2", "w2_err", "Vega", 46028.0, 171.787, 0.047, "WISE/WISE.W2"),
    _bp("AllWISE", "W3", "w3", "w3_err", "Vega", 115608.0, 31.674, mid_ir_av_coefficient("AllWISE", "W3"), "WISE/WISE.W3"),
    _bp("AllWISE", "W4", "w4", "w4_err", "Vega", 220883.0, 8.363, mid_ir_av_coefficient("AllWISE", "W4"), "WISE/WISE.W4"),
    _bp(
        "IPHAS", "Halpha", "iphas_ha_mag", None, "Vega", 6568.0, 2950.0, 0.815, "INT/IPHAS.Ha",
        response_kind="emission_line_filter",
        fit_policy="diagnostic_only",
        policy_flags=("emission_line", "diagnostic_only"),
    ),
    # Added catalog families
    _bp("Gaia GSPC", "SDSS_u", "gspc_sdss_u", "gspc_sdss_u_err", "AB", 3543.0, None, 1.579, "SLOAN/SDSS.u", True),
    _bp("Gaia GSPC", "SDSS_g", "gspc_sdss_g", "gspc_sdss_g_err", "AB", 4770.0, None, 1.199, "SLOAN/SDSS.g", True),
    _bp("Gaia GSPC", "SDSS_r", "gspc_sdss_r", "gspc_sdss_r_err", "AB", 6231.0, None, 0.858, "SLOAN/SDSS.r", True),
    _bp("Gaia GSPC", "SDSS_i", "gspc_sdss_i", "gspc_sdss_i_err", "AB", 7625.0, None, 0.639, "SLOAN/SDSS.i", True),
    _bp("Gaia GSPC", "SDSS_z", "gspc_sdss_z", "gspc_sdss_z_err", "AB", 9134.0, None, 0.453, "SLOAN/SDSS.z", True),
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
    _bp("SkyMapper", "u", "skymapper_u", "skymapper_u_err", "AB", 3490.0, None, 1.579, "SkyMapper/SkyMapper.u"),
    _bp("SkyMapper", "v", "skymapper_v", "skymapper_v_err", "AB", 3840.0, None, 1.420, "SkyMapper/SkyMapper.v"),
    _bp("SkyMapper", "g", "skymapper_g", "skymapper_g_err", "AB", 5100.0, None, 1.199, "SkyMapper/SkyMapper.g"),
    _bp("SkyMapper", "r", "skymapper_r", "skymapper_r_err", "AB", 6170.0, None, 0.858, "SkyMapper/SkyMapper.r"),
    _bp("SkyMapper", "i", "skymapper_i", "skymapper_i_err", "AB", 7790.0, None, 0.639, "SkyMapper/SkyMapper.i"),
    _bp("SkyMapper", "z", "skymapper_z", "skymapper_z_err", "AB", 9160.0, None, 0.453, "SkyMapper/SkyMapper.z"),
    _bp("DES", "g", "des_g", "des_g_err", "AB", 4770.0, None, 1.199, "CTIO/DECam.g"),
    _bp("DES", "r", "des_r", "des_r_err", "AB", 6400.0, None, 0.858, "CTIO/DECam.r"),
    _bp("DES", "i", "des_i", "des_i_err", "AB", 7830.0, None, 0.639, "CTIO/DECam.i"),
    _bp("DES", "z", "des_z", "des_z_err", "AB", 9170.0, None, 0.453, "CTIO/DECam.z"),
    _bp("DES", "Y", "des_y", "des_y_err", "AB", 9890.0, None, 0.385, "CTIO/DECam.Y"),
    _bp("DECaPS", "g", "decaps_g", "decaps_g_err", "AB", 4770.0, None, 1.199, "CTIO/DECam.g"),
    _bp("DECaPS", "r", "decaps_r", "decaps_r_err", "AB", 6400.0, None, 0.858, "CTIO/DECam.r"),
    _bp("DECaPS", "i", "decaps_i", "decaps_i_err", "AB", 7830.0, None, 0.639, "CTIO/DECam.i"),
    _bp("DECaPS", "z", "decaps_z", "decaps_z_err", "AB", 9170.0, None, 0.453, "CTIO/DECam.z"),
    _bp("DECaPS", "Y", "decaps_y", "decaps_y_err", "AB", 9890.0, None, 0.385, "CTIO/DECam.Y"),
    _bp("UKIDSS", "Y", "ukidss_y", "ukidss_y_err", "Vega", 10300.0, 2026.0, 0.38, "UKIRT/UKIDSS.Y"),
    _bp("UKIDSS", "J", "ukidss_j", "ukidss_j_err", "Vega", 12500.0, 1530.0, 0.282, "UKIRT/UKIDSS.J"),
    _bp("UKIDSS", "H", "ukidss_h", "ukidss_h_err", "Vega", 16350.0, 1019.0, 0.175, "UKIRT/UKIDSS.H"),
    _bp("UKIDSS", "K", "ukidss_k", "ukidss_k_err", "Vega", 22000.0, 631.0, 0.112, "UKIRT/UKIDSS.K"),
    _bp("VISTA/VVV", "Z", "vista_z", "vista_z_err", "Vega", 8780.0, 2217.0, 0.453, "Paranal/VISTA.Z"),
    _bp("VISTA/VVV", "Y", "vista_y", "vista_y_err", "Vega", 10200.0, 2026.0, 0.385, "Paranal/VISTA.Y"),
    _bp("VISTA/VVV", "J", "vista_j", "vista_j_err", "Vega", 12500.0, 1530.0, 0.282, "Paranal/VISTA.J"),
    _bp("VISTA/VVV", "H", "vista_h", "vista_h_err", "Vega", 16350.0, 1019.0, 0.175, "Paranal/VISTA.H"),
    _bp("VISTA/VVV", "Ks", "vista_ks", "vista_ks_err", "Vega", 21500.0, 631.0, 0.112, "Paranal/VISTA.Ks"),
    _bp("VISTA/VHS", "Y", None, None, "Vega", 10200.0, 2026.0, 0.385, "Paranal/VISTA.Y"),
    _bp("VISTA/VHS", "J", None, None, "Vega", 12500.0, 1530.0, 0.282, "Paranal/VISTA.J"),
    _bp("VISTA/VHS", "H", None, None, "Vega", 16350.0, 1019.0, 0.175, "Paranal/VISTA.H"),
    _bp("VISTA/VHS", "Ks", None, None, "Vega", 21500.0, 631.0, 0.112, "Paranal/VISTA.Ks"),
    _bp("VISTA/VIKING", "Z", None, None, "Vega", 8780.0, 2217.0, 0.453, "Paranal/VISTA.Z"),
    _bp("VISTA/VIKING", "Y", None, None, "Vega", 10200.0, 2026.0, 0.385, "Paranal/VISTA.Y"),
    _bp("VISTA/VIKING", "J", None, None, "Vega", 12500.0, 1530.0, 0.282, "Paranal/VISTA.J"),
    _bp("VISTA/VIKING", "H", None, None, "Vega", 16350.0, 1019.0, 0.175, "Paranal/VISTA.H"),
    _bp("VISTA/VIKING", "Ks", None, None, "Vega", 21500.0, 631.0, 0.112, "Paranal/VISTA.Ks"),
    _bp("CatWISE2020", "W1", None, None, "Vega", 33526.0, 309.540, 0.061, "WISE/WISE.W1", confusion_risk=True),
    _bp("CatWISE2020", "W2", None, None, "Vega", 46028.0, 171.787, 0.047, "WISE/WISE.W2", confusion_risk=True),
    _bp(
        "NOIRLab NSC DR2", "u", None, None, "AB", 3543.0, None, 1.579, None,
        response_kind="mixed_unknown", fit_policy="diagnostic_only",
        policy_flags=("mixed_instrument_mean", "instrument_provenance_required", "nsc_exact_unavailable", "diagnostic_only", "bad_quality"),
    ),
    _bp(
        "NOIRLab NSC DR2", "g", None, None, "AB", 4770.0, None, 1.199, None,
        response_kind="mixed_unknown", fit_policy="diagnostic_only",
        policy_flags=("mixed_instrument_mean", "instrument_provenance_required", "nsc_exact_unavailable", "diagnostic_only", "bad_quality"),
    ),
    _bp(
        "NOIRLab NSC DR2", "r", None, None, "AB", 6400.0, None, 0.858, None,
        response_kind="mixed_unknown", fit_policy="diagnostic_only",
        policy_flags=("mixed_instrument_mean", "instrument_provenance_required", "nsc_exact_unavailable", "diagnostic_only", "bad_quality"),
    ),
    _bp(
        "NOIRLab NSC DR2", "i", None, None, "AB", 7830.0, None, 0.639, None,
        response_kind="mixed_unknown", fit_policy="diagnostic_only",
        policy_flags=("mixed_instrument_mean", "instrument_provenance_required", "nsc_exact_unavailable", "diagnostic_only", "bad_quality"),
    ),
    _bp(
        "NOIRLab NSC DR2", "z", None, None, "AB", 9170.0, None, 0.453, None,
        response_kind="mixed_unknown", fit_policy="diagnostic_only",
        policy_flags=("mixed_instrument_mean", "instrument_provenance_required", "nsc_exact_unavailable", "diagnostic_only", "bad_quality"),
    ),
    _bp(
        "NOIRLab NSC DR2", "Y", None, None, "AB", 9890.0, None, 0.385, None,
        response_kind="mixed_unknown", fit_policy="diagnostic_only",
        policy_flags=("mixed_instrument_mean", "instrument_provenance_required", "nsc_exact_unavailable", "diagnostic_only", "bad_quality"),
    ),
    _bp(
        "NOIRLab NSC DR2", "VR", None, None, "AB", 6300.0, None, 0.90, None,
        response_kind="mixed_unknown", fit_policy="diagnostic_only",
        policy_flags=("mixed_instrument_mean", "instrument_provenance_required", "nsc_exact_unavailable", "diagnostic_only", "bad_quality"),
    ),
    _bp("Swift/UVOT", "UVW2", None, None, "AB", 1928.0, None, 2.90, "Swift/UVOT.UVW2"),
    _bp("Swift/UVOT", "UVM2", None, None, "AB", 2246.0, None, 2.85, "Swift/UVOT.UVM2"),
    _bp("Swift/UVOT", "UVW1", None, None, "AB", 2600.0, None, 2.55, "Swift/UVOT.UVW1"),
    _bp("Swift/UVOT", "U", None, None, "AB", 3465.0, None, 1.58, "Swift/UVOT.U"),
    _bp("Swift/UVOT", "B", None, None, "AB", 4392.0, None, 1.32, "Swift/UVOT.B"),
    _bp("Swift/UVOT", "V", None, None, "AB", 5468.0, None, 1.00, "Swift/UVOT.V"),
    _bp("XMM-OM", "UVW2", None, None, "AB", 2120.0, None, 2.90, "XMM/OM.UVW2"),
    _bp("XMM-OM", "UVM2", None, None, "AB", 2310.0, None, 2.85, "XMM/OM.UVM2"),
    _bp("XMM-OM", "UVW1", None, None, "AB", 2910.0, None, 2.55, "XMM/OM.UVW1"),
    _bp("XMM-OM", "U", None, None, "AB", 3440.0, None, 1.58, "XMM/OM.U"),
    _bp("XMM-OM", "B", None, None, "AB", 4500.0, None, 1.32, "XMM/OM.B"),
    _bp("XMM-OM", "V", None, None, "AB", 5430.0, None, 1.00, "XMM/OM.V"),
    # II/386 APER MAG3 values are Vega magnitudes.  SVO's OmegaCAM profiles
    # contain the physical filter+CCD response, but not the complete
    # atmosphere+telescope response of each VPHAS+ exposure, so the broad
    # bands are explicitly fitted as natural-system proxies with a 0.04-mag
    # systematic floor.  The scalar zero points below are offline fallbacks;
    # cached response-matched SVO/Vega calibrations take precedence.
    _bp(
        "VPHAS+", "u", "vphas_u", "vphas_u_err", "Vega", 3543.0, 1550.81, 1.579,
        "Paranal/OmegaCAM.u_SDSS", response_kind="filter_ccd_natural_system_proxy",
        fit_policy="photosphere_proxy", systematic_floor_mag=0.04,
        policy_flags=("natural_system_proxy", "filter_plus_ccd_response", "systematic_floor_mag=0.04"),
    ),
    _bp(
        "VPHAS+", "g", "vphas_g", "vphas_g_err", "Vega", 4770.0, 3960.53, 1.199,
        "Paranal/OmegaCAM.g_SDSS", response_kind="filter_ccd_natural_system_proxy",
        fit_policy="photosphere_proxy", systematic_floor_mag=0.04,
        policy_flags=("natural_system_proxy", "filter_plus_ccd_response", "systematic_floor_mag=0.04"),
    ),
    _bp(
        "VPHAS+", "r", "vphas_r", "vphas_r_err", "Vega", 6231.0, 3094.68, 0.858,
        "Paranal/OmegaCAM.r_SDSS", response_kind="filter_ccd_natural_system_proxy",
        fit_policy="photosphere_proxy", systematic_floor_mag=0.04,
        policy_flags=("natural_system_proxy", "filter_plus_ccd_response", "systematic_floor_mag=0.04"),
    ),
    _bp(
        "VPHAS+", "i", "vphas_i", "vphas_i_err", "Vega", 7625.0, 2563.84, 0.639,
        "Paranal/OmegaCAM.i_SDSS", response_kind="filter_ccd_natural_system_proxy",
        fit_policy="photosphere_proxy", systematic_floor_mag=0.04,
        policy_flags=("natural_system_proxy", "filter_plus_ccd_response", "systematic_floor_mag=0.04"),
    ),
    _bp(
        "VPHAS+", "Halpha", "vphas_ha", "vphas_ha_err", "Vega", 6568.0, 2950.0, 0.815,
        "Paranal/OmegaCAM.Halpha", response_kind="emission_line_filter",
        fit_policy="diagnostic_only",
        policy_flags=("emission_line", "diagnostic_only"),
    ),
    _bp("Spitzer SEIP", "IRAC1", "spitzer_irac1", "spitzer_irac1_err", "Jy", 35500.0, None, mid_ir_av_coefficient("Spitzer SEIP", "IRAC1"), "Spitzer/IRAC.I1", confusion_risk=True),
    _bp("Spitzer SEIP", "IRAC2", "spitzer_irac2", "spitzer_irac2_err", "Jy", 44930.0, None, mid_ir_av_coefficient("Spitzer SEIP", "IRAC2"), "Spitzer/IRAC.I2", confusion_risk=True),
    _bp("Spitzer SEIP", "IRAC3", "spitzer_irac3", "spitzer_irac3_err", "Jy", 57310.0, None, mid_ir_av_coefficient("Spitzer SEIP", "IRAC3"), "Spitzer/IRAC.I3", confusion_risk=True),
    _bp("Spitzer SEIP", "IRAC4", "spitzer_irac4", "spitzer_irac4_err", "Jy", 78720.0, None, mid_ir_av_coefficient("Spitzer SEIP", "IRAC4"), "Spitzer/IRAC.I4", confusion_risk=True),
    _bp("Spitzer SEIP", "MIPS24", "spitzer_mips24", "spitzer_mips24_err", "Jy", 236750.0, None, mid_ir_av_coefficient("Spitzer SEIP", "MIPS24"), "Spitzer/MIPS.24mu", confusion_risk=True),
    _bp("AKARI", "S9W", "akari_s9w", "akari_s9w_err", "Jy", 90000.0, None, mid_ir_av_coefficient("AKARI", "S9W"), "AKARI/IRC.S9W", confusion_risk=True),
    _bp("AKARI", "L18W", "akari_l18w", "akari_l18w_err", "Jy", 180000.0, None, mid_ir_av_coefficient("AKARI", "L18W"), "AKARI/IRC.L18W", confusion_risk=True),
    _bp("AKARI", "N60", "akari_n60", "akari_n60_err", "Jy", 650000.0, None, 0.0, "AKARI/FIS.N60", confusion_risk=True),
    _bp("AKARI", "WIDE-S", "akari_wide_s", "akari_wide_s_err", "Jy", 900000.0, None, 0.0, "AKARI/FIS.WIDE-S", confusion_risk=True),
    _bp("AKARI", "WIDE-L", "akari_wide_l", "akari_wide_l_err", "Jy", 1400000.0, None, 0.0, "AKARI/FIS.WIDE-L", confusion_risk=True),
    _bp("AKARI", "N160", "akari_n160", "akari_n160_err", "Jy", 1600000.0, None, 0.0, "AKARI/FIS.N160", confusion_risk=True),
    _bp("IRAS", "12", "iras_12", "iras_12_err", "Jy", 120000.0, None, mid_ir_av_coefficient("IRAS", "12"), "IRAS/IRAS.12mu", confusion_risk=True),
    _bp("IRAS", "25", "iras_25", "iras_25_err", "Jy", 250000.0, None, mid_ir_av_coefficient("IRAS", "25"), "IRAS/IRAS.25mu", confusion_risk=True),
    _bp("IRAS", "60", "iras_60", "iras_60_err", "Jy", 600000.0, None, 0.0, "IRAS/IRAS.60mu", confusion_risk=True),
    _bp("IRAS", "100", "iras_100", "iras_100_err", "Jy", 1000000.0, None, 0.0, "IRAS/IRAS.100mu", confusion_risk=True),
    _bp("Herschel", "PACS70", "herschel_pacs70", "herschel_pacs70_err", "Jy", 700000.0, None, 0.0, "Herschel/Pacs.blue", confusion_risk=True),
    _bp("Herschel", "PACS100", "herschel_pacs100", "herschel_pacs100_err", "Jy", 1000000.0, None, 0.0, "Herschel/Pacs.green", confusion_risk=True),
    _bp("Herschel", "PACS160", "herschel_pacs160", "herschel_pacs160_err", "Jy", 1600000.0, None, 0.0, "Herschel/Pacs.red", confusion_risk=True),
    _bp("Herschel", "SPIRE250", None, None, "Jy", 2500000.0, None, 0.0, "Herschel/SPIRE.PSW", confusion_risk=True),
    _bp("Herschel", "SPIRE350", None, None, "Jy", 3500000.0, None, 0.0, "Herschel/SPIRE.PMW", confusion_risk=True),
    _bp("Herschel", "SPIRE500", None, None, "Jy", 5000000.0, None, 0.0, "Herschel/SPIRE.PLW", confusion_risk=True),
    _bp("APEX", "SABOCA350", None, None, "Jy", 3500000.0, None, 0.0, None, confusion_risk=True),
    _bp("APEX", "LABOCA870", None, None, "Jy", 8700000.0, None, 0.0, None, confusion_risk=True),
]:
    SED_BANDPASSES[_band_key(_b.source, _b.band)] = _b

# NSC DR2 object means mix three cameras and therefore retain the diagnostic
# registrations above.  Measurement/exposure joins can resolve these exact
# physical responses.  The internal keys intentionally coexist with the
# source/band object-mean keys so ``malca sed-bandpasses --all`` also validates
# every response needed by exact NSC rows.
NSC_INSTRUMENT_ALIASES = {
    "c4d": "c4d",
    "decam": "c4d",
    "k4m": "k4m",
    "mosaic3": "k4m",
    "mosaic": "k4m",
    "ksb": "ksb",
    "90prime": "ksb",
    "90primecam": "ksb",
    "bass": "ksb",
}
_NSC_EXACT_FILTER_IDS = {
    "c4d": {
        "u": "CTIO/DECam.u",
        "g": "CTIO/DECam.g",
        "r": "CTIO/DECam.r",
        "i": "CTIO/DECam.i",
        "z": "CTIO/DECam.z",
        "Y": "CTIO/DECam.Y",
        "VR": "CTIO/DECam.VR_filter",
    },
    "k4m": {"z": "KPNO/MOSAIC.zd_DECam"},
    "ksb": {"g": "BOK/BASS.g", "r": "BOK/BASS.r"},
}
NSC_INSTRUMENT_BANDPASSES: dict[tuple[str, str], SedBandpass] = {}
for _instrument, _filters in _NSC_EXACT_FILTER_IDS.items():
    for _band, _filter_id in _filters.items():
        _mean_bp = SED_BANDPASSES[_band_key("NOIRLab NSC DR2", _band)]
        _filter_only = _filter_id.endswith("_filter") or (_instrument, _band.casefold()) == ("k4m", "z")
        _exact_bp = _bp(
            "NOIRLab NSC DR2",
            _band,
            None,
            None,
            "AB",
            _mean_bp.lambda_eff_angstrom,
            None,
            _mean_bp.av_coeff,
            _filter_id,
            response_kind="filter_only_proxy" if _filter_only else "instrument_response",
            fit_policy="diagnostic_only" if _filter_only else "photosphere",
            policy_flags=(
                "nsc_measurement_level",
                *(("filter_only_response", "diagnostic_only") if _filter_only else ()),
            ),
        )
        NSC_INSTRUMENT_BANDPASSES[(_instrument, _band.casefold())] = _exact_bp
        SED_BANDPASSES[f"__nsc_exact__:{_instrument}:{_band.casefold()}"] = _exact_bp

PAYLOAD_BANDPASSES = tuple(
    b for b in SED_BANDPASSES.values()
    if b.mag_col
    and (
        b.source
        in {"Gaia DR3", "GALEX", "APASS", "2MASS", "AllWISE", "IPHAS", "VPHAS+"}
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
    "VISTA/VHS": "#ff8f70",
    "VISTA/VIKING": "#ff4f81",
    "AllWISE": "#e25555",
    "CatWISE2020": "#c24141",
    "NOIRLab NSC DR2": "#84cc16",
    "Gaia XP": "#c4b5fd",
    "Swift/UVOT": "#38bdf8",
    "XMM-OM": "#0284c7",
    "Spitzer SEIP": "#d95f02",
    "AKARI": "#b2182b",
    "IRAS": "#8b0000",
    "Herschel": "#6a3d9a",
    "APEX": "#5b2c83",
    # Midpoint between Gaia DR3 (#d7b43c) and 2MASS (#ffb347), matching
    # IPHAS H-alpha's intermediate wavelength in the source-ordered SED.
    "IPHAS": "#ebb442",
    "VPHAS+": "#e377c2",
}

SED_SOURCE_LABELS = {
    "payload": "Payload",
    "gaia_gspc": "Gaia GSPC",
    "gaia_xp": "Gaia XP",
    "galex": "GALEX",
    "catwise": "CatWISE2020",
    "nsc": "NOIRLab NSC DR2",
    "ps1": "PS1",
    "sdss": "SDSS",
    "skymapper": "SkyMapper",
    "des": "DES",
    "decaps": "DECaPS",
    "ukidss": "UKIDSS",
    "vista": "VISTA/VVV",
    "vhs": "VISTA/VHS",
    "viking": "VISTA/VIKING",
    "vphas": "VPHAS+",
    "swift_uvot": "Swift/UVOT",
    "xmm_om": "XMM-OM",
    "allwise": "AllWISE IRSA",
    "spitzer": "Spitzer",
    "akari": "AKARI",
    "iras": "IRAS",
    "herschel": "Herschel",
    "apex_laboca": "APEX/LABOCA",
    "apex_saboca": "APEX/SABOCA",
}

_PAYLOAD_SED_SOURCES = {"gaia dr3", "apass", "2mass", "allwise", "iphas", "vphas+"}
_SED_ROW_SOURCE_TO_KEY = {
    "gaia gspc": "gaia_gspc",
    "gaia xp": "gaia_xp",
    "galex": "galex",
    "catwise2020": "catwise",
    "catwise": "catwise",
    "noirlab nsc dr2": "nsc",
    "nsc dr2": "nsc",
    "pan-starrs": "ps1",
    "pan starrs": "ps1",
    "ps1": "ps1",
    "sdss": "sdss",
    "skymapper": "skymapper",
    "des": "des",
    "decaps": "decaps",
    "ukidss": "ukidss",
    "vista/vvv": "vista",
    "vvv": "vista",
    "vista/vhs": "vhs",
    "vhs": "vhs",
    "vista/viking": "viking",
    "viking": "viking",
    "vphas+": "vphas",
    "vphas": "vphas",
    "swift/uvot": "swift_uvot",
    "swift uvot": "swift_uvot",
    "xmm-om": "xmm_om",
    "xmm om": "xmm_om",
    "spitzer seip": "spitzer",
    "spitzer": "spitzer",
    "akari": "akari",
    "iras": "iras",
    "herschel": "herschel",
    "apex": "apex_laboca",
}


def _safe_float(value: object) -> float | None:
    if np.ma.is_masked(value):
        return None
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


def _clean_text(value: object) -> str:
    """Return a stripped scalar string, treating pandas missing values as empty."""
    try:
        if value is None or pd.isna(value):
            return ""
    except (TypeError, ValueError):
        if value is None:
            return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>", "--"} else text


def _normalize_integer_id(value: object) -> str | None:
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        if value is None:
            return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return None
        return str(int(value))
    text = str(value).strip()
    if not text or text.lower() in {"nan", "<na>", "none", "--"}:
        return None
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    if "e" in text.lower():
        try:
            return str(int(Decimal(text)))
        except (InvalidOperation, ValueError):
            return text
    return text


def _to_bool_int(value: object) -> int:
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "true", "t", "yes", "y"} else 0
    try:
        if value is None or pd.isna(value):
            return 0
    except (TypeError, ValueError):
        if value is None:
            return 0
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


def bandpass_for(source: str, band: str, instrument: str | None = None) -> SedBandpass | None:
    if _band_key(source, "").startswith("noirlab nsc dr2:") and _clean_text(instrument):
        instrument_key = re.sub(r"[^a-z0-9]+", "", _clean_text(instrument).casefold())
        canonical_instrument = NSC_INSTRUMENT_ALIASES.get(instrument_key, instrument_key)
        exact = NSC_INSTRUMENT_BANDPASSES.get((canonical_instrument, str(band).casefold()))
        if exact is not None:
            return exact
    return SED_BANDPASSES.get(_band_key(source, band))


def _first_payload_float(payload: Mapping[str, object], names: Iterable[str]) -> float | None:
    for name in names:
        value = _safe_float(payload.get(name))
        if value is not None:
            return value
    return None


def apass_b_policy_flags(payload: Mapping[str, object]) -> tuple[str, ...]:
    """Return conservative APASS-B red-leak policy flags.

    The AAVSO warning is defined using Johnson B minus Cousins Ic.  Sloan i is
    deliberately not substituted: doing so would turn a documented threshold
    into an undocumented color transform.
    """
    b_mag = _first_payload_float(payload, ("apass_b", "Bmag", "b_mag"))
    ic_mag = _first_payload_float(
        payload,
        ("cousins_i", "cousins_ic", "ic_mag", "i_c_mag", "Icmag", "Ic"),
    )
    if b_mag is None or ic_mag is None:
        return ("apass_b_red_leak_unassessed",)
    color = b_mag - ic_mag
    if color > APASS_B_RED_LEAK_COLOR_THRESHOLD:
        # ``bad_quality`` keeps the current bandpass fitter from using the point;
        # the specific flag lets v3 report the actual policy reason.
        return ("apass_b_red_leak_likely", "bad_quality")
    return ("apass_b_red_leak_screened",)


def _bandpass_policy_flags(
    bandpass: SedBandpass,
    payload: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    flags = list(bandpass.policy_flags)
    if bandpass.source == "APASS" and bandpass.band == "B":
        flags.extend(apass_b_policy_flags(payload if payload is not None else {}))
    if bandpass.mag_system.strip().upper() == "JY":
        flags.extend(("native_flux_density", "mission_calibration_required", "diagnostic_only"))
    return tuple(dict.fromkeys(flag for flag in flags if flag))


def _response_zero_point_jy(
    bandpass: SedBandpass,
    response_zero_points_jy: Mapping[object, float] | None,
) -> tuple[float | None, str]:
    system = bandpass.mag_system.strip().upper()
    if system == "AB":
        return 3631.0, "AB_definition"
    if system == "JY":
        return None, "native_catalog_flux"
    if response_zero_points_jy:
        keys: tuple[object, ...] = (
            (str(bandpass.svo_filter_id or ""), bandpass.mag_system),
            (str(bandpass.svo_filter_id or ""), system),
            str(bandpass.svo_filter_id or ""),
            _band_key(bandpass.source, bandpass.band),
        )
        for key in keys:
            value = _safe_float(response_zero_points_jy.get(key))
            if value is not None and value > 0:
                return value, "response_calibration"
    return bandpass.fnu_zero_jy, "legacy_registry_zero_point"


_REGISTERED_RESPONSE_CACHE_MAXSIZE = 256
_REGISTERED_RESPONSE_CACHE: OrderedDict[tuple[str, str], object] = OrderedDict()
_REGISTERED_RESPONSE_CACHE_LOCK = RLock()


def _clear_cached_registered_responses() -> None:
    """Clear process-local positive response hits (primarily for refresh/tests)."""
    with _REGISTERED_RESPONSE_CACHE_LOCK:
        _REGISTERED_RESPONSE_CACHE.clear()


def _load_cached_registered_response(filter_id: str, mag_system: str) -> object | None:
    """Load one response from MALCA's local cache without network access.

    The import stays lazy so basic review/payload loading does not require the
    synthetic-photometry stack until a registered SVO response is actually
    present.  The bounded process-local cache prevents one NPZ decompression
    per candidate during large backfills; callers that populate the disk cache
    in the same process need no explicit invalidation after a previous miss:
    misses are deliberately not memoized.  This helper must never call
    ``fetch_filter_response``.
    """
    key = (str(filter_id), str(mag_system))
    with _REGISTERED_RESPONSE_CACHE_LOCK:
        cached = _REGISTERED_RESPONSE_CACHE.get(key)
        if cached is not None:
            _REGISTERED_RESPONSE_CACHE.move_to_end(key)
            return cached
    try:
        from malca.enrichment.synthetic_photometry import load_cached_filter_response

        response = load_cached_filter_response(*key)
    except Exception:
        return None
    if response is None:
        return None
    with _REGISTERED_RESPONSE_CACHE_LOCK:
        _REGISTERED_RESPONSE_CACHE[key] = response
        _REGISTERED_RESPONSE_CACHE.move_to_end(key)
        while len(_REGISTERED_RESPONSE_CACHE) > _REGISTERED_RESPONSE_CACHE_MAXSIZE:
            _REGISTERED_RESPONSE_CACHE.popitem(last=False)
    return response


def _cached_bandpass_response_metadata(bandpass: SedBandpass) -> dict[str, object]:
    """Return cache-only response/calibration metadata for a registered band.

    A Vega calibration is accepted only from a cache view explicitly resolved
    as Vega.  If the physical curve exists only under an old AB cache entry, it
    may still supply the response hash and pivot wavelength, but its 3631-Jy
    scalar is never reused as a Vega zero point.
    """
    system = bandpass.mag_system.strip().upper()
    filter_id = str(bandpass.svo_filter_id or "").strip()
    if not filter_id:
        return {}

    calibrated_response = _load_cached_registered_response(filter_id, bandpass.mag_system)
    response = calibrated_response
    if response is None:
        # Throughput is calibration-independent.  A legacy cache can therefore
        # still supply pivot/hash metadata while its wrong-system scalar is
        # deliberately ignored.
        response = _load_cached_registered_response(filter_id, "")
    if response is None:
        return {}

    try:
        from malca.enrichment.synthetic_photometry import (
            calibration_for_response,
            response_pivot_wavelength_angstrom,
        )

        pivot = _safe_float(response_pivot_wavelength_angstrom(response))
    except Exception:
        pivot = None

    metadata: dict[str, object] = {
        "response_hash": str(getattr(response, "response_hash", "") or ""),
    }
    if pivot is not None and system != "JY":
        metadata.update(
            {
                "lambda_pivot_angstrom": pivot,
                "plot_lambda_angstrom": pivot,
                "plot_lambda_kind": "response_pivot",
            }
        )

    if system == "JY":
        # Direct-Jy products retain the mission reference wavelength, but the
        # physical throughput hash is still needed to detect a refreshed
        # response before reusing an old fit normalization.
        return metadata

    calibration_response = calibrated_response
    if system == "VEGA" and (
        calibration_response is None
        or str(getattr(calibration_response, "mag_system", "") or "").strip().upper() != "VEGA"
        or _safe_float(getattr(calibration_response, "zero_point_jy", None)) is None
    ):
        calibration_response = None
    if calibration_response is None and system == "AB":
        calibration_response = response
    if calibration_response is not None:
        try:
            calibration = calibration_for_response(calibration_response, bandpass.mag_system)
            metadata.update(
                {
                    "response_zero_point_jy": calibration.zero_point_jy,
                    "calibration_id": calibration.calibration_id,
                    "calibration_hash": calibration.calibration_hash,
                }
            )
        except Exception:
            pass
    return metadata


def _wavelength_semantics(
    item: Mapping[str, object] | pd.Series | None,
    bandpass: SedBandpass,
) -> dict[str, object]:
    values = item if item is not None else {}
    nominal = (
        _safe_float(values.get("lambda_nominal_angstrom"))
        or _safe_float(values.get("nominal_wavelength_angstrom"))
        or bandpass.lambda_nominal_angstrom
    )
    pivot = (
        _safe_float(values.get("lambda_pivot_angstrom"))
        or _safe_float(values.get("pivot_wavelength_angstrom"))
        or bandpass.lambda_pivot_angstrom
    )
    if bandpass.mag_system.strip().upper() == "JY" and bandpass.lambda_reference_angstrom is not None:
        # Direct-Jy catalogs quote a monochromatic flux at the mission-defined
        # reference wavelength.  SVO WavelengthRef describes its response
        # calibration and must not move that catalog observable on the plot.
        reference = bandpass.lambda_reference_angstrom
    else:
        reference = (
            _safe_float(values.get("lambda_reference_angstrom"))
            or _safe_float(values.get("reference_wavelength_angstrom"))
            or bandpass.lambda_reference_angstrom
        )
    isophotal = _safe_float(values.get("lambda_isophotal_angstrom")) or bandpass.lambda_isophotal_angstrom
    explicit_plot = _safe_float(values.get("plot_lambda_angstrom")) or _safe_float(values.get("plot_wavelength_angstrom"))
    explicit_kind = str(values.get("plot_lambda_kind") or values.get("plot_wavelength_kind") or "").strip()
    if bandpass.mag_system.strip().upper() == "JY" and reference is not None:
        # A quoted monochromatic flux belongs at the mission reference
        # wavelength even when an upstream response lookup supplied its own
        # generic plotting wavelength.
        plot_wave = reference
        plot_kind = "mission_reference"
    elif pivot is not None:
        plot_wave = pivot
        plot_kind = "response_pivot"
    elif explicit_plot is not None:
        plot_wave = explicit_plot
        plot_kind = explicit_kind or "explicit"
    else:
        plot_wave = _safe_float(values.get("lambda_eff_angstrom")) or bandpass.lambda_eff_angstrom
        plot_kind = explicit_kind or bandpass.plot_lambda_kind
    return {
        "plot_lambda_angstrom": float(plot_wave),
        "plot_lambda_kind": str(plot_kind),
        "lambda_nominal_angstrom": nominal,
        "lambda_pivot_angstrom": pivot,
        "lambda_reference_angstrom": reference,
        "lambda_isophotal_angstrom": isophotal,
    }


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


def flux_nu_jy_from_flux_lambda(flux_lambda: float, lambda_angstrom: float) -> float | None:
    flux = _safe_float(flux_lambda)
    lam = _safe_float(lambda_angstrom)
    if flux is None or flux <= 0 or lam is None or lam <= 0:
        return None
    wavelength = lam * u.AA
    return (flux * u.erg / u.s / u.cm**2 / u.AA).to(
        u.Jy,
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
    mag: float | None,
    mag_err: float | None,
    distance_pc: float | None,
    av: float | None,
    dereddened: bool,
    sep_arcsec: float | None = None,
    quality_flags: str = "",
    is_upper_limit: bool = False,
    flux_nu_jy: float | None = None,
    flux_nu_jy_err: float | None = None,
    response_zero_points_jy: Mapping[object, float] | None = None,
    wavelength_metadata: Mapping[str, object] | pd.Series | None = None,
    policy_payload: Mapping[str, object] | None = None,
    prefer_input_flux_nu: bool = False,
) -> dict | None:
    system = bandpass.mag_system.strip().upper()
    metadata = dict(wavelength_metadata) if wavelength_metadata is not None else {}
    response_metadata_resolved = bool(metadata.get("_response_metadata_resolved"))
    cached_metadata: dict[str, object] = {}
    if system != "JY" and bandpass.svo_filter_id and not response_metadata_resolved:
        cached_metadata = _cached_bandpass_response_metadata(bandpass)
        # Current cached response state is authoritative for response-derived
        # quantities.  Native catalog provenance in ``metadata`` remains
        # untouched.
        for key in (
            "lambda_pivot_angstrom",
            "plot_lambda_angstrom",
            "plot_lambda_kind",
            "response_hash",
            "calibration_id",
            "calibration_hash",
        ):
            value = cached_metadata.get(key)
            if value is not None and str(value).strip():
                metadata[key] = value

    resolved_zero_points: dict[object, float] = {}
    cached_zero_point = _safe_float(cached_metadata.get("response_zero_point_jy"))
    if cached_zero_point is not None and cached_zero_point > 0:
        resolved_zero_points[(str(bandpass.svo_filter_id or ""), bandpass.mag_system)] = cached_zero_point
        resolved_zero_points[(str(bandpass.svo_filter_id or ""), system)] = cached_zero_point
        resolved_zero_points[str(bandpass.svo_filter_id or "")] = cached_zero_point
        resolved_zero_points[_band_key(bandpass.source, bandpass.band)] = cached_zero_point
    resolved_zero_points.update(dict(response_zero_points_jy or {}))

    flags = [x for x in str(quality_flags or "").split(";") if x]
    flags.extend(_bandpass_policy_flags(bandpass, policy_payload))
    if bandpass.confusion_risk and "confusion_risk" not in flags:
        flags.append("confusion_risk")
    effective_mag = _safe_float(mag)
    input_flux_scale = 1.0
    can_deredden = dereddened and av is not None and bandpass.av_coeff is not None
    if dereddened:
        if can_deredden:
            input_flux_scale = 10.0 ** (0.4 * float(av) * float(bandpass.av_coeff))
            flags.append("ism_corrected")
        else:
            flags.append("no_extinction_coeff")
    if can_deredden and system != "JY" and effective_mag is not None:
        effective_mag = effective_mag - float(av) * float(bandpass.av_coeff)
    calibration_source = "native_catalog_flux"
    if system == "JY":
        flux_nu = _safe_float(flux_nu_jy)
        if flux_nu is None:
            flux_nu = effective_mag
        if flux_nu is not None:
            flux_nu = float(flux_nu) * input_flux_scale
        effective_mag = None
    else:
        # An explicitly supplied response view (e.g. the fitter's in-memory
        # response map) takes precedence over the lazy disk-cache view.
        zero_point, calibration_source = _response_zero_point_jy(
            bandpass,
            response_zero_points_jy,
        )
        if calibration_source != "response_calibration":
            zero_point, calibration_source = _response_zero_point_jy(
                bandpass,
                resolved_zero_points,
            )
        if bandpass.svo_filter_id and not response_metadata_resolved and not cached_metadata:
            flags.append("legacy_response_metadata_fallback")
        if system == "VEGA" and calibration_source == "legacy_registry_zero_point":
            flags.append("legacy_vega_zero_point_fallback")
        if prefer_input_flux_nu and _safe_float(flux_nu_jy) is not None:
            flux_nu = _safe_float(flux_nu_jy)
            calibration_source = "stored_canonical_normalization"
            # A stored v3 value is the exact observed normalization.  Apply
            # extinction as a multiplicative correction to that value instead
            # of rebuilding it from a potentially different Vega zero point.
            if can_deredden:
                flux_nu = float(flux_nu) * input_flux_scale
        elif effective_mag is not None and zero_point is not None and zero_point > 0:
            flux_nu = float(zero_point) * 10.0 ** (-0.4 * effective_mag)
        else:
            flux_nu = _safe_float(flux_nu_jy)
            if flux_nu is not None and can_deredden:
                flux_nu = float(flux_nu) * input_flux_scale
    if flux_nu is None:
        return None
    wavelength = _wavelength_semantics(metadata, bandpass)
    plot_wave = float(wavelength["plot_lambda_angstrom"])
    flux_lambda = flux_lambda_from_flux_nu_jy(flux_nu, plot_wave)
    if flux_lambda is None:
        return None
    flux_nu_err = (
        _safe_float(flux_nu_jy_err)
        if system == "JY" or prefer_input_flux_nu or effective_mag is None
        else None
    )
    if flux_nu_err is not None and input_flux_scale != 1.0:
        flux_nu_err *= input_flux_scale
    flux_lambda_err = None
    lambda_l_err = None
    merr = None if system == "JY" else _safe_float(mag_err)
    if merr is not None and merr <= 0:
        merr = None
    if flux_nu_err is not None and flux_nu_err <= 0:
        flux_nu_err = None
    if system != "JY" and merr is not None:
        frac = math.log(10.0) * 0.4 * merr
        flux_nu_err = abs(flux_nu) * frac
    if flux_nu_err is not None and flux_nu > 0:
        flux_lambda_err = abs(flux_lambda) * abs(flux_nu_err / flux_nu)

    lambda_l = lambda_l_lambda_from_flux_lambda(
        flux_lambda,
        plot_wave,
        distance_pc,
    )
    if lambda_l is not None and flux_lambda_err is not None and flux_lambda > 0:
        lambda_l_err = abs(lambda_l) * abs(flux_lambda_err / flux_lambda)

    row = {
        "candidate_id": str(candidate_id),
        "source": bandpass.source,
        "band": bandpass.band,
        "mag": effective_mag,
        "mag_err": merr,
        "mag_system": bandpass.mag_system,
        "lambda_eff_angstrom": plot_wave,
        "flux_lambda": float(flux_lambda),
        "flux_lambda_err": flux_lambda_err,
        "lambda_l_lambda": lambda_l,
        "lambda_l_lambda_err": lambda_l_err,
        "flux_nu_jy": float(flux_nu),
        "flux_nu_jy_err": flux_nu_err,
        "observed_flux_nu_jy": float(flux_nu),
        "observed_flux_nu_jy_err": flux_nu_err,
        "sep_arcsec": sep_arcsec,
        "is_synthetic": int(bandpass.is_synthetic),
        "is_upper_limit": int(is_upper_limit),
        "quality_flags": ";".join(sorted(set(flags))),
        "svo_filter_id": bandpass.svo_filter_id,
        "av_coeff": bandpass.av_coeff,
        **wavelength,
        "response_kind": bandpass.response_kind,
        "fit_policy": bandpass.fit_policy,
        "systematic_floor_mag": bandpass.systematic_floor_mag,
        "observable_kind": (
            "quoted_fnu" if system == "JY" else "ab_mag" if system == "AB" else "vega_mag"
        ),
        "passband_fidelity": (
            "standardized_proxy"
            if bandpass.response_kind == "standardized_system_proxy"
            else "natural_system_proxy"
            if bandpass.response_kind == "filter_ccd_natural_system_proxy"
            else "emission_line_diagnostic"
            if bandpass.response_kind == "emission_line_filter"
            else "mixed_unknown"
            if bandpass.response_kind == "mixed_unknown"
            else "exact"
        ),
        "native_flux_unit": "Jy" if system == "JY" else "magnitude",
        "calibration_source": calibration_source,
        "calibration_id": metadata.get("calibration_id") or (
            f"{bandpass.svo_filter_id or _band_key(bandpass.source, bandpass.band)}:"
            f"{'quoted_fnu' if system == 'JY' else system.lower()}"
        ),
        "calibration_hash": metadata.get("calibration_hash"),
        "response_hash": metadata.get("response_hash"),
        "normalization_version": CANONICAL_SED_NORMALIZATION_VERSION,
    }
    # Measurement identity must be constructed from the native observation,
    # not merely candidate/source/band.  In particular, retain catalog,
    # instrument, exposure, and epoch provenance before hashing so repeated
    # visits remain distinct canonical measurements.
    provenance_aliases = {
        "measurement_id": ("measurement_id",),
        "catalog_release": ("catalog_release", "release", "data_release"),
        "source_object_id": ("source_object_id", "catalog_object_id", "object_id"),
        "catalog_measurement_id": (
            "catalog_measurement_id",
            "source_measurement_id",
            "observation_id",
        ),
        "instrument": ("instrument", "camera"),
        "exposure_id": ("exposure_id", "visit_id"),
        "epoch_mjd": ("epoch_mjd", "observation_mjd", "mjd"),
        "correlation_group": ("correlation_group",),
        "provenance_json": ("provenance_json", "measurement_provenance_json"),
    }
    for provenance_column, aliases in provenance_aliases.items():
        for alias in aliases:
            value = metadata.get(alias)
            if value is None:
                continue
            try:
                missing = pd.isna(value)
            except (TypeError, ValueError):
                missing = False
            if isinstance(missing, (bool, np.bool_)) and missing:
                continue
            if str(value).strip():
                row[provenance_column] = value
                break
    try:
        from malca.review.sed_storage import make_sed_measurement_id

        if not _clean_text(row.get("measurement_id")):
            row["measurement_id"] = make_sed_measurement_id(
                {
                    **row,
                    "catalog": row.get("source"),
                    "release": row.get("catalog_release"),
                    "catalog_object_id": row.get("source_object_id"),
                }
            )
    except (ImportError, TypeError, ValueError):
        row.setdefault("measurement_id", None)
    return row


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
        quality_flags = ""
        sep_arcsec = None
        is_upper_limit = False
        if bandpass.source == "AllWISE" and bandpass.band.upper() in {"W1", "W2", "W3", "W4"}:
            band_number = int(bandpass.band[1])
            quality_parts = []
            for output_name, payload_name in (
                ("qph", "allwise_ph_qual"),
                ("ccf", "allwise_cc_flags"),
                ("ex", "allwise_ext_flg"),
                ("nb", "allwise_nb"),
                ("na", "allwise_na"),
                ("var", "allwise_var_flg"),
                (f"snr{band_number}", f"allwise_w{band_number}_snr"),
                (f"chi2W{band_number}", f"allwise_w{band_number}_rchi2"),
                (f"sat{band_number}", f"allwise_w{band_number}_sat"),
                (f"nW{band_number}", f"allwise_w{band_number}_ndet"),
                (f"mW{band_number}", f"allwise_w{band_number}_nframe"),
            ):
                value = payload.get(payload_name)
                try:
                    missing = value is None or pd.isna(value)
                except (TypeError, ValueError):
                    missing = value is None
                if not missing and str(value).strip() not in {"", "nan", "--"}:
                    quality_parts.append(f"{output_name}={str(value).strip()}")
            quality_flags = ";".join(quality_parts)
            sep_arcsec = _safe_float(payload.get("allwise_sep_arcsec"))
            qph = str(payload.get("allwise_ph_qual") or "")
            if len(qph) >= band_number:
                is_upper_limit = qph[band_number - 1].upper() == "U"
        row = _row_from_bandpass(
            candidate_id=cid,
            bandpass=bandpass,
            mag=mag,
            mag_err=mag_err,
            distance_pc=distance_pc,
            av=av,
            dereddened=dereddened,
            sep_arcsec=sep_arcsec,
            quality_flags=quality_flags,
            is_upper_limit=is_upper_limit,
            policy_payload=payload,
        )
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows, columns=CANONICAL_SED_COLUMNS)


def prepare_canonical_sed_measurements(
    rows: pd.DataFrame | Iterable[dict] | None,
    *,
    payload: Mapping[str, object] | None = None,
    candidate_id: str = "",
    extinction_mode: str = "observed",
    response_zero_points_jy: Mapping[object, float] | None = None,
) -> pd.DataFrame:
    """Return the single canonical measurement representation for v3 consumers.

    Both the review plot and the fitter can call this function.  Supplying
    response-derived Vega zero points makes their magnitude-to-flux conversion
    identical; without them, the explicitly labelled legacy registry fallback
    remains available for old/offline review databases.
    """
    if rows is None:
        return pd.DataFrame(columns=CANONICAL_SED_COLUMNS)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=CANONICAL_SED_COLUMNS)

    payload_dict: Mapping[str, object] = payload if payload is not None else {}
    mode = str(extinction_mode or "observed").strip().lower()
    dereddened = mode in {"corrected", "ism-corrected", "ism_corrected", "dereddened"}
    av = extinction_av_from_payload(dict(payload_dict))
    distance_pc = distance_pc_from_payload(dict(payload_dict))
    cid = str(candidate_id or payload_dict.get("candidate_id") or payload_dict.get("asas_sn_id") or "unknown")

    out: list[dict] = []
    for _, item in frame.iterrows():
        source = str(item.get("source") or "").strip()
        band = str(item.get("band") or "").strip()
        bp = bandpass_for(source, band, instrument=_clean_text(item.get("instrument")) or None)
        registered_bandpass = bp is not None
        if bp is None:
            legacy_wave = _safe_float(item.get("lambda_eff_angstrom"))
            explicit_wave = _safe_float(item.get("plot_lambda_angstrom")) or _safe_float(item.get("plot_wavelength_angstrom"))
            if legacy_wave is None and explicit_wave is None:
                continue
            mag_system = str(item.get("mag_system") or "AB")
            bp = _bp(
                source or "Catalog",
                band or "?",
                None,
                None,
                mag_system,
                explicit_wave or legacy_wave or 1.0,
                None if mag_system.upper() == "AB" else _safe_float(item.get("fnu_zero_jy")),
                _safe_float(item.get("av_coeff")),
                str(item.get("svo_filter_id") or "") or None,
                bool(_to_bool_int(item.get("is_synthetic", False))),
                "confusion_risk" in str(item.get("quality_flags") or ""),
                plot_lambda_kind=str(item.get("plot_lambda_kind") or item.get("plot_wavelength_kind") or "legacy_external_value"),
                lambda_nominal_angstrom=(
                    _safe_float(item.get("lambda_nominal_angstrom"))
                    or _safe_float(item.get("nominal_wavelength_angstrom"))
                ),
                lambda_pivot_angstrom=(
                    _safe_float(item.get("lambda_pivot_angstrom"))
                    or _safe_float(item.get("pivot_wavelength_angstrom"))
                ),
                lambda_reference_angstrom=(
                    _safe_float(item.get("lambda_reference_angstrom"))
                    or _safe_float(item.get("reference_wavelength_angstrom"))
                ),
                lambda_isophotal_angstrom=_safe_float(item.get("lambda_isophotal_angstrom")),
                response_kind=str(item.get("response_kind") or "unknown_external_response"),
                fit_policy=str(item.get("fit_policy") or "diagnostic_until_registered"),
            )

        system = bp.mag_system.strip().upper()
        mag = _safe_float(item.get("mag"))
        observed_flux_nu = _safe_float(item.get("observed_flux_nu_jy"))
        flux_nu = (
            observed_flux_nu
            if observed_flux_nu is not None
            else _safe_float(item.get("flux_nu_jy"))
        )
        observed_flux_nu_err = _safe_float(item.get("observed_flux_nu_jy_err"))
        flux_nu_err = (
            observed_flux_nu_err
            if observed_flux_nu_err is not None
            else _safe_float(item.get("flux_nu_jy_err"))
        )
        input_flux_only = mag is None and flux_nu is not None
        if flux_nu is None:
            input_flux_lambda = _safe_float(item.get("flux_lambda"))
            if input_flux_lambda is not None:
                plot_wave = float(_wavelength_semantics(item, bp)["plot_lambda_angstrom"])
                flux_nu = flux_nu_jy_from_flux_lambda(input_flux_lambda, plot_wave)
                input_flux_only = flux_nu is not None
                input_flux_lambda_err = _safe_float(item.get("flux_lambda_err"))
                if input_flux_lambda_err is not None:
                    flux_nu_err = flux_nu_jy_from_flux_lambda(input_flux_lambda_err, plot_wave)
        normalization_version = _clean_text(item.get("normalization_version"))
        stored_normalization_snapshot = (
            bool(_clean_text(item.get("measurement_id")))
            and normalization_version.startswith("sed-measurement-v")
            and bool(_clean_text(item.get("normalization_hash")))
            and observed_flux_nu is not None
            and observed_flux_nu > 0
            and (
                _safe_float(item.get("plot_lambda_angstrom")) is not None
                or _safe_float(item.get("lambda_eff_angstrom")) is not None
            )
        )
        stored_canonical_flux = (
            observed_flux_nu is not None
            and stored_normalization_snapshot
            and not response_zero_points_jy
        )
        if mag is None and flux_nu is not None and system != "JY":
            zero_point, _ = _response_zero_point_jy(bp, response_zero_points_jy)
            if zero_point is not None and zero_point > 0:
                mag = -2.5 * math.log10(flux_nu / zero_point)
        if mag is None and flux_nu is None:
            continue
        wavelength_metadata = dict(item)
        if stored_normalization_snapshot:
            # A complete ledger normalization is an immutable snapshot.  Do
            # not mix its flux/wavelength/hash identity with a newer response
            # cache view; cache drift is checked separately before fit-point
            # substitution.
            wavelength_metadata["_response_metadata_resolved"] = True
        row = _row_from_bandpass(
            candidate_id=cid,
            bandpass=bp,
            mag=mag,
            mag_err=_safe_float(item.get("mag_err")),
            distance_pc=distance_pc,
            av=av,
            dereddened=dereddened,
            sep_arcsec=_safe_float(item.get("sep_arcsec")),
            quality_flags=str(item.get("quality_flags") or ""),
            is_upper_limit=bool(_to_bool_int(item.get("is_upper_limit", False))),
            flux_nu_jy=flux_nu,
            flux_nu_jy_err=flux_nu_err,
            response_zero_points_jy=response_zero_points_jy,
            wavelength_metadata=wavelength_metadata,
            policy_payload=payload_dict,
            prefer_input_flux_nu=input_flux_only or stored_canonical_flux,
        )
        if row is not None:
            # The review path without a response map must display an immutable
            # normalization verbatim.  The fitter, however, supplies a
            # response-derived zero point specifically to build a new
            # response-calibrated product; copying the v3 flux/wavelength
            # fields back here would combine that new identity with old
            # numeric values.
            if stored_normalization_snapshot and not dereddened and not response_zero_points_jy:
                immutable_fields = (
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
                    "observed_flux_nu_jy",
                    "observed_flux_nu_jy_err",
                    "svo_filter_id",
                    "plot_lambda_angstrom",
                    "plot_lambda_kind",
                    "lambda_nominal_angstrom",
                    "lambda_pivot_angstrom",
                    "lambda_reference_angstrom",
                    "lambda_isophotal_angstrom",
                    "calibration_id",
                    "calibration_hash",
                    "response_hash",
                    "normalization_version",
                    "normalization_hash",
                )
                for field in immutable_fields:
                    value = item.get(field)
                    try:
                        missing = value is None or pd.isna(value)
                    except (TypeError, ValueError):
                        missing = value is None
                    if not missing and str(value).strip():
                        row[field] = value
            for provenance_column in (
                "measurement_id",
                "normalization_version",
                "catalog_release",
                "source_object_id",
                "catalog_measurement_id",
                "instrument",
                "exposure_id",
                "epoch_mjd",
                "correlation_group",
                "provenance_json",
                "native_value",
                "native_error",
                "native_unit",
                "calibration_id",
                "calibration_hash",
                "response_hash",
                "normalization_hash",
                "passband_fidelity",
                "observable_kind",
            ):
                # Registered source/band semantics and freshly resolved cache
                # identities are authoritative over legacy rows that may have
                # been normalized with an old magnitude system or response.
                if registered_bandpass and provenance_column in {"passband_fidelity", "observable_kind"}:
                    continue
                if provenance_column in {
                    "calibration_id",
                    "calibration_hash",
                    "response_hash",
                } and _clean_text(row.get(provenance_column)):
                    continue
                value = item.get(provenance_column)
                try:
                    missing = value is None or pd.isna(value)
                except (TypeError, ValueError):
                    missing = value is None
                if not missing and str(value).strip():
                    row[provenance_column] = value
            out.append(row)
    return pd.DataFrame(out, columns=CANONICAL_SED_COLUMNS)


def prepare_sed_measurement_frame(
    rows: pd.DataFrame | Iterable[dict] | None,
    *,
    payload: Mapping[str, object] | None = None,
    candidate_id: str = "",
    extinction_mode: str = "observed",
    responses: Mapping[object, object] | None = None,
    response_zero_points_jy: Mapping[object, float] | None = None,
) -> pd.DataFrame:
    """Shared plot/fitter canonicalization entry point for a measurement frame."""
    zero_points: dict[object, float] = dict(response_zero_points_jy or {})
    for key, response in (responses or {}).items():
        zero_point = _safe_float(getattr(response, "zero_point_jy", None))
        if zero_point is not None and zero_point > 0:
            zero_points[key] = zero_point
            filter_id = str(getattr(response, "filter_id", "") or "")
            if filter_id:
                zero_points[filter_id] = zero_point
    return prepare_canonical_sed_measurements(
        rows,
        payload=payload,
        candidate_id=candidate_id,
        extinction_mode=extinction_mode,
        response_zero_points_jy=zero_points,
    )


def prepare_sed_measurement_row(
    row: Mapping[str, object] | pd.Series,
    *,
    payload: Mapping[str, object] | None = None,
    candidate_id: str = "",
    extinction_mode: str = "observed",
    response: object | None = None,
    zero_point_jy: float | None = None,
) -> dict[str, object] | None:
    """Canonicalize one catalog row for identical use by plotting and fitting."""
    item = dict(row)
    source = str(item.get("source") or "")
    band = str(item.get("band") or "")
    bandpass = bandpass_for(source, band)
    zero_points: dict[object, float] = {}
    resolved_zero = _safe_float(zero_point_jy)
    if resolved_zero is None and response is not None:
        resolved_zero = _safe_float(getattr(response, "zero_point_jy", None))
    if resolved_zero is not None and resolved_zero > 0 and bandpass is not None:
        zero_points[_band_key(source, band)] = resolved_zero
        if bandpass.svo_filter_id:
            zero_points[bandpass.svo_filter_id] = resolved_zero
    if response is not None and _safe_float(item.get("lambda_reference_angstrom")) is None:
        reference = _safe_float(getattr(response, "wavelength_ref_angstrom", None))
        if reference is not None:
            item["lambda_reference_angstrom"] = reference
    if response is not None:
        item["_response_metadata_resolved"] = True
        item["response_hash"] = str(getattr(response, "response_hash", "") or "")
        if _safe_float(item.get("lambda_pivot_angstrom")) is None:
            try:
                from malca.enrichment.synthetic_photometry import response_pivot_wavelength_angstrom

                item["lambda_pivot_angstrom"] = response_pivot_wavelength_angstrom(response)
            except Exception:
                pass
        if bandpass is not None and bandpass.mag_system.strip().upper() != "JY":
            try:
                from malca.enrichment.synthetic_photometry import calibration_for_response

                calibration = calibration_for_response(response, bandpass.mag_system)
                item["calibration_id"] = calibration.calibration_id
                item["calibration_hash"] = calibration.calibration_hash
            except Exception:
                pass
    prepared = prepare_sed_measurement_frame(
        [item],
        payload=payload,
        candidate_id=candidate_id,
        extinction_mode=extinction_mode,
        response_zero_points_jy=zero_points,
    )
    if prepared.empty:
        return None
    return prepared.iloc[0].to_dict()


def normalize_external_sed_rows(
    rows: pd.DataFrame | Iterable[dict] | None,
    *,
    payload: dict,
    candidate_id: str,
    extinction_mode: str = "observed",
) -> pd.DataFrame:
    """Backward-compatible legacy-column view of canonical SED measurements."""
    canonical = prepare_sed_measurement_frame(
        rows,
        payload=payload,
        candidate_id=candidate_id,
        extinction_mode=extinction_mode,
    )
    return canonical.reindex(columns=SED_COLUMNS)


def _concat_sed_frames(*frames: pd.DataFrame) -> pd.DataFrame:
    """Concatenate SED tables without pandas empty-column FutureWarnings."""
    parts = [frame for frame in frames if frame is not None and not frame.empty]
    if not parts:
        return pd.DataFrame()
    if len(parts) == 1:
        return parts[0].copy()
    columns = list(dict.fromkeys(column for part in parts for column in part.columns))
    aligned = [part.reindex(columns=columns) for part in parts]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        return pd.concat(aligned, ignore_index=True, sort=False)


def build_sed_dataframe(
    payload: dict,
    *,
    candidate_id: str | None = None,
    external_rows: pd.DataFrame | Iterable[dict] | None = None,
    extinction_mode: str = "observed",
) -> pd.DataFrame:
    cid = str(candidate_id or payload.get("candidate_id") or payload.get("asas_sn_id") or "")
    payload_rows = rows_from_payload(payload, candidate_id=cid, extinction_mode=extinction_mode)
    payload_rows = payload_rows.copy()
    payload_rows["_stored_identity_rank"] = 0
    external = prepare_sed_measurement_frame(
        external_rows,
        payload=payload,
        candidate_id=cid,
        extinction_mode=extinction_mode,
    )
    external = external.copy()
    external["_stored_identity_rank"] = 1
    combined = _concat_sed_frames(payload_rows, external)
    if combined.empty:
        return pd.DataFrame(columns=CANONICAL_SED_COLUMNS)
    # Avoid payload/persisted duplicates without collapsing real multi-epoch
    # measurements that share a catalog and band.  A persisted immutable
    # normalization wins over the payload reconstruction for the same native
    # measurement ID; otherwise the exact fit hash can be discarded before the
    # substitution check even runs.
    combined["_measurement_key"] = _sed_measurement_dedupe_key(combined)
    combined = combined.sort_values(
        "_stored_identity_rank", ascending=False, kind="stable"
    ).drop_duplicates(subset=["_measurement_key"], keep="first")
    combined["_source_rank"] = combined["source"].astype(str).map(
        lambda s: 1 if s == "Gaia GSPC" else 0
    )
    combined = combined.sort_values(
        ["lambda_eff_angstrom", "_source_rank", "source", "band"], kind="stable"
    ).drop(columns=["_source_rank", "_measurement_key", "_stored_identity_rank"])
    return combined.reset_index(drop=True)


def _theme(theme: str | None) -> dict[str, str]:
    mode = str(theme or "black").strip().lower()
    if mode == "white":
        return {"paper": "#ffffff", "plot": "#ffffff", "font": "#1c2733", "grid": "rgba(104,128,149,0.18)", "muted": "#5a6b7b"}
    if mode == "gray":
        return {"paper": "#2e3440", "plot": "#2e3440", "font": "#d8dee9", "grid": "rgba(129,161,193,0.15)", "muted": "#aab6c7"}
    return {"paper": "#0d0d0d", "plot": "#0d0d0d", "font": "#dce8f2", "grid": "rgba(96,116,130,0.22)", "muted": "#9fb6cb"}


def _fit_normalization_identity(row: Mapping[str, object] | pd.Series) -> tuple[str, str, str]:
    return (
        _clean_text(row.get("measurement_id")),
        _clean_text(row.get("normalization_version")),
        _clean_text(row.get("normalization_hash")),
    )


def _current_cached_response_hash(row: Mapping[str, object] | pd.Series) -> str:
    bandpass = bandpass_for(
        _clean_text(row.get("source")),
        _clean_text(row.get("band")),
        instrument=_clean_text(row.get("instrument")) or None,
    )
    if bandpass is None or not bandpass.svo_filter_id:
        return ""
    return _clean_text(_cached_bandpass_response_metadata(bandpass).get("response_hash"))


def _log_axis_range_from_data(values: Iterable[object], *, pad_dex: float = 0.08, min_span_dex: float = 0.35) -> list[float] | None:
    """Return a Plotly log-axis range based only on positive finite data values."""
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        return None
    logs = np.log10(arr)
    lo = float(np.nanmin(logs))
    hi = float(np.nanmax(logs))
    center = 0.5 * (lo + hi)
    span = max(hi - lo, min_span_dex)
    half = 0.5 * span + pad_dex
    return [center - half, center + half]


def build_sed_figure(
    payload: dict,
    *,
    candidate_id: str | None = None,
    external_rows: pd.DataFrame | Iterable[dict] | None = None,
    model_curve_rows: pd.DataFrame | Iterable[dict] | None = None,
    model_fit_rows: pd.DataFrame | Iterable[dict] | None = None,
    model_point_rows: pd.DataFrame | Iterable[dict] | None = None,
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
            sed_df = pd.DataFrame(columns=CANONICAL_SED_COLUMNS + ["sed_mode"])
        else:
            sed_df = pd.DataFrame(
                [*observed.to_dict("records"), *corrected.to_dict("records")],
                columns=CANONICAL_SED_COLUMNS + ["sed_mode"],
            )
    else:
        sed_df = build_sed_dataframe(payload, candidate_id=candidate_id, external_rows=external_rows, extinction_mode=mode)
        sed_df["sed_mode"] = "ISM-corrected" if mode in {"corrected", "ism-corrected", "ism_corrected", "dereddened"} else "Observed"

    point_model_df = pd.DataFrame(model_point_rows) if model_point_rows is not None else pd.DataFrame()
    stale_fit_warning = (
        "Stored CK fit inputs do not match the current SED measurement/normalization identities; "
        "fit-point substitution is disabled."
    )
    intrinsic_fit_warning = (
        "Intrinsic CK comparison is hidden because exact current-fit bandpass extinction "
        "ratios are unavailable for one or more displayed points."
    )
    fit_identity_exact = False
    intrinsic_ratio_complete = False
    if sed_df.empty and not point_model_df.empty:
        # The previous fit can outlive every one of its catalog inputs.  This
        # is still a stale fit even though there is no current row over which
        # the substitution loop could discover the mismatch.
        warnings.append(stale_fit_warning)
    if not sed_df.empty and not point_model_df.empty:
        # A current fit stores the exact canonical observed value it consumed.
        # Reuse it only when the immutable measurement and normalization
        # recipe both match.  Source/band is not an identity: an instrument or
        # multi-epoch catalog can legitimately contain several such rows.
        point_identity_lookup: dict[tuple[str, str, str], pd.Series] = {}
        point_identity_counts: dict[tuple[str, str, str], int] = {}
        for _, point in point_model_df.iterrows():
            identity = _fit_normalization_identity(point)
            point_identity_counts[identity] = point_identity_counts.get(identity, 0) + 1
            point_identity_lookup[identity] = point
        point_identity_lookup = {
            identity: point
            for identity, point in point_identity_lookup.items()
            if point_identity_counts.get(identity) == 1 and all(identity)
        }
        distance_pc = distance_pc_from_payload(payload)
        current_identities = {
            _fit_normalization_identity(row)
            for _, row in sed_df.iterrows()
        }
        point_identities = {
            _fit_normalization_identity(row)
            for _, row in point_model_df.iterrows()
        }
        incomplete_identity = any(not all(identity) for identity in current_identities | point_identities)
        stale_fit_inputs = (
            incomplete_identity
            or len(point_identities) != len(point_model_df)
            or current_identities != point_identities
        )
        if not stale_fit_inputs:
            # The normalization hash binds the stored response, but a local
            # cache refresh happens after that hash was minted.  Compare the
            # current throughput hash separately and never combine a new pivot
            # or response with an old normalization identity.
            for _, row in sed_df.drop_duplicates(
                subset=["measurement_id", "normalization_version", "normalization_hash"]
            ).iterrows():
                identity = _fit_normalization_identity(row)
                point = point_identity_lookup.get(identity)
                if point is None:
                    stale_fit_inputs = True
                    break
                row_response_hash = _clean_text(row.get("response_hash"))
                point_response_hash = _clean_text(point.get("response_hash"))
                if row_response_hash != point_response_hash:
                    stale_fit_inputs = True
                    break
                current_response_hash = _current_cached_response_hash(row)
                if (
                    current_response_hash
                    and row_response_hash
                    and current_response_hash != row_response_hash
                ):
                    stale_fit_inputs = True
                    break
        fit_identity_exact = not stale_fit_inputs
        corrected_expected = 0
        corrected_applied = 0
        for idx, row in sed_df.iterrows():
            if stale_fit_inputs:
                continue
            identity = _fit_normalization_identity(row)
            point = point_identity_lookup.get(identity)
            if point is None:
                continue
            observed_fnu = _safe_float(point.get("observed_flux_nu_jy"))
            if observed_fnu is None or observed_fnu <= 0:
                continue
            observed_fnu_err = _safe_float(point.get("observed_flux_nu_jy_err"))
            display_fnu = observed_fnu
            display_fnu_err = observed_fnu_err
            row_mode = _clean_text(row.get("sed_mode")).lower()
            if row_mode != "observed":
                corrected_expected += 1
                model_observed_fnu = _safe_float(point.get("model_flux_nu_jy"))
                model_intrinsic_fnu = _safe_float(point.get("model_flux_nu_jy_intrinsic"))
                if (
                    model_observed_fnu is None
                    or model_observed_fnu <= 0
                    or model_intrinsic_fnu is None
                    or model_intrinsic_fnu <= 0
                ):
                    # Leave the scalar payload-Av display in place, but it is
                    # not comparable to the intrinsic CK curve and is labelled
                    # as such below.
                    continue
                correction_ratio = model_intrinsic_fnu / model_observed_fnu
                display_fnu = observed_fnu * correction_ratio
                display_fnu_err = (
                    observed_fnu_err * correction_ratio
                    if observed_fnu_err is not None
                    else None
                )
                corrected_applied += 1
            plot_lambda = (
                _safe_float(point.get("plot_lambda_angstrom"))
                or _safe_float(point.get("lambda_pivot_angstrom"))
                or _safe_float(point.get("lambda_reference_angstrom"))
                or _safe_float(row.get("plot_lambda_angstrom"))
                or _safe_float(row.get("lambda_eff_angstrom"))
            )
            if plot_lambda is None or plot_lambda <= 0:
                continue
            display_flam = flux_lambda_from_flux_nu_jy(display_fnu, plot_lambda)
            if display_flam is None:
                continue
            sed_df.at[idx, "flux_nu_jy"] = display_fnu
            sed_df.at[idx, "observed_flux_nu_jy"] = observed_fnu
            sed_df.at[idx, "flux_nu_jy_err"] = display_fnu_err
            sed_df.at[idx, "observed_flux_nu_jy_err"] = observed_fnu_err
            sed_df.at[idx, "plot_lambda_angstrom"] = plot_lambda
            sed_df.at[idx, "plot_lambda_kind"] = str(point.get("plot_lambda_kind") or "fit_canonical")
            sed_df.at[idx, "lambda_eff_angstrom"] = plot_lambda
            sed_df.at[idx, "flux_lambda"] = display_flam
            display_flam_err = (
                abs(display_flam * display_fnu_err / display_fnu)
                if display_fnu_err is not None and display_fnu_err > 0
                else None
            )
            sed_df.at[idx, "flux_lambda_err"] = display_flam_err
            lambda_l = lambda_l_lambda_from_flux_lambda(display_flam, plot_lambda, distance_pc)
            sed_df.at[idx, "lambda_l_lambda"] = lambda_l
            sed_df.at[idx, "lambda_l_lambda_err"] = (
                abs(lambda_l * display_fnu_err / display_fnu)
                if lambda_l is not None and display_fnu_err is not None and display_fnu_err > 0
                else None
            )
        intrinsic_ratio_complete = (
            fit_identity_exact
            and corrected_expected > 0
            and corrected_applied == corrected_expected
        )
        if stale_fit_inputs:
            warnings.append(stale_fit_warning)

    corrected_modes = {"corrected", "ism-corrected", "ism_corrected", "dereddened"}
    intrinsic_requested = mode == "both" or mode in corrected_modes
    if intrinsic_requested and point_model_df.empty:
        intrinsic_ratio_complete = False

    spec = _theme(theme)
    fig = go.Figure()
    y_col = "lambda_l_lambda"
    if sed_df.empty or sed_df[y_col].isna().all():
        y_col = "flux_lambda"
        if distance_pc_from_payload(payload) is None:
            warnings.append("No distance available; plotting flux density instead of luminosity.")

    if sed_df.empty:
        fig.add_annotation(text="No SED photometry available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        phot_x = np.array([], dtype=float)
        phot_y = np.array([], dtype=float)
    else:
        plot_df = sed_df.copy()
        plot_x = pd.to_numeric(plot_df["plot_lambda_angstrom"], errors="coerce")
        legacy_x = pd.to_numeric(plot_df["lambda_eff_angstrom"], errors="coerce")
        plot_df["x"] = plot_x.where(np.isfinite(plot_x) & (plot_x > 0), legacy_x)
        y_scale = LSUN_ERG_S if y_col == "lambda_l_lambda" else 1.0
        plot_df["y"] = pd.to_numeric(plot_df[y_col], errors="coerce") / y_scale
        plot_df = plot_df[np.isfinite(plot_df["x"]) & np.isfinite(plot_df["y"]) & (plot_df["x"] > 0) & (plot_df["y"] > 0)]
        phot_x = plot_df["x"].to_numpy(dtype=float)
        phot_y = plot_df["y"].to_numpy(dtype=float)
        for (mode_name, source), grp in plot_df.groupby(["sed_mode", "source"], dropna=False):
            color = SOURCE_COLORS.get(str(source), "#bbbbbb")
            native_flux_group = bool(
                grp["mag_system"].fillna("").astype(str).str.strip().str.upper().eq("JY").all()
            )
            is_sampled_spectrum = bool(
                len(grp) > 1
                and grp["quality_flags"].fillna("").astype(str).str.contains("correlated_spectrum", regex=False).all()
            )
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
            opacity = 1.0
            y_err_col = "lambda_l_lambda_err" if y_col == "lambda_l_lambda" else "flux_lambda_err"
            y_err = (pd.to_numeric(grp.get(y_err_col), errors="coerce") / y_scale) if y_err_col in grp.columns else None
            show_y_err = bool(y_err is not None and np.isfinite(y_err).any())
            fig.add_trace(go.Scatter(
                x=grp["x"],
                y=grp["y"],
                mode="lines" if is_sampled_spectrum else "markers",
                name=f"{source} ({mode_name})" if mode == "both" else str(source),
                marker=dict(size=10, color=color, symbol=symbols, opacity=opacity, line=dict(width=1, color=spec["font"])),
                line=dict(color=color, width=1.2),
                error_y=dict(type="data", array=y_err, visible=show_y_err, thickness=0.8),
                customdata=np.column_stack([
                    grp["band"].astype(str),
                    pd.to_numeric(grp["mag"], errors="coerce"),
                    grp["mag_system"].astype(str),
                    grp["quality_flags"].astype(str),
                    pd.to_numeric(grp["flux_nu_jy"], errors="coerce"),
                ]),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "band: %{customdata[0]}<br>"
                    "lambda: %{x:.5g} A<br>"
                    + (
                        "F_nu: %{customdata[4]:.4g} Jy<br>"
                        if native_flux_group
                        else "mag: %{customdata[1]:.4g} %{customdata[2]}<br>"
                    )
                    + ("lambda L_lambda: %{y:.4e} Lsun<br>" if y_col == "lambda_l_lambda" else "F_lambda: %{y:.4e}<br>")
                    + "flags: %{customdata[3]}<extra></extra>"
                ),
            ))

    if model_curve_rows is not None:
        model_df = pd.DataFrame(model_curve_rows)
    else:
        model_df = pd.DataFrame()
    if model_fit_rows is not None:
        fit_df = pd.DataFrame(model_fit_rows)
    else:
        fit_df = pd.DataFrame()
    if (
        intrinsic_requested
        and (not point_model_df.empty or not model_df.empty or not fit_df.empty)
        and not intrinsic_ratio_complete
    ):
        warnings.append(intrinsic_fit_warning)
    fit_version = str(fit_df.iloc[0].get("fit_version") or "") if not fit_df.empty else ""
    bandpass_v2 = fit_version.startswith("ck04-bandpass-v")
    model_is_comparable = (
        bandpass_v2 and (mode not in corrected_modes or intrinsic_ratio_complete)
    )
    model_color = "#111827" if str(theme or "black").strip().lower() == "white" else "#f8fafc"

    if not fit_df.empty:
        fit_status = str(fit_df.iloc[0].get("status") or "").strip()
        fit_warning = str(fit_df.iloc[0].get("warning") or "").strip()
        teff = _safe_float(fit_df.iloc[0].get("teff_k"))
        chi2_nu = _safe_float(fit_df.iloc[0].get("reduced_chi2"))
        n_fit = _safe_float(fit_df.iloc[0].get("n_fit_points"))
        av_fit = _safe_float(fit_df.iloc[0].get("av_fit"))
        if fit_status == "ok":
            details = []
            if teff is not None:
                details.append(f"Teff={teff:.0f} K")
            if chi2_nu is not None:
                details.append(f"chi2_nu={chi2_nu:.2g}")
            if n_fit is not None:
                details.append(f"n={int(n_fit)}")
            if av_fit is not None:
                details.append(f"Av={av_fit:.2f}")
            if model_is_comparable:
                warnings.append("CK fit: " + ", ".join(details) if details else "CK fit available.")
            elif intrinsic_requested and bandpass_v2:
                summary = ", ".join(details) if details else "available"
                warnings.append(
                    f"CK fit is available ({summary}), but its intrinsic overlay is withheld "
                    "until exact bandpass correction ratios are available."
                )
            else:
                summary = ", ".join(details) if details else "available"
                warnings.append(f"CK fit is dereddened ({summary}); switch to ISM-corrected or Both to compare it with the points.")
        elif fit_status:
            warnings.append(f"CK fit: {fit_status}" + (f" ({fit_warning})" if fit_warning else ""))

    if model_is_comparable and not model_df.empty and "wavelength_angstrom" in model_df.columns:
        curve_modes = ["observed"] if mode not in corrected_modes and mode != "both" else ["intrinsic"]
        if mode == "both":
            curve_modes = ["observed", "intrinsic"]
        for curve_mode in curve_modes:
            if curve_mode == "intrinsic" and not intrinsic_ratio_complete:
                continue
            if bandpass_v2:
                curve_col = (
                    f"lambda_l_lambda_{curve_mode}"
                    if y_col == "lambda_l_lambda"
                    else f"flux_lambda_{curve_mode}"
                )
            else:
                curve_col = y_col
                if curve_mode == "observed":
                    continue
            if curve_col not in model_df.columns:
                continue
            curve = model_df.copy()
            curve["x"] = pd.to_numeric(curve["wavelength_angstrom"], errors="coerce")
            y_scale = LSUN_ERG_S if y_col == "lambda_l_lambda" else 1.0
            curve["y"] = pd.to_numeric(curve[curve_col], errors="coerce") / y_scale
            curve = curve[np.isfinite(curve["x"]) & np.isfinite(curve["y"]) & (curve["x"] > 0) & (curve["y"] > 0)]
            if curve.empty:
                continue
            curve = curve.sort_values("x")
            teff = _safe_float(curve.iloc[0].get("teff_k"))
            name = f"Castelli/Kurucz {curve_mode} fit"
            if teff is not None:
                name += f" ({teff:.0f} K)"
            fig.add_trace(go.Scatter(
                x=curve["x"],
                y=curve["y"],
                mode="lines",
                name=name,
                line=dict(
                    color=model_color,
                    width=2.2,
                    dash="dash" if curve_mode == "intrinsic" and mode == "both" else "solid",
                ),
                opacity=0.95,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "lambda: %{x:.5g} A<br>"
                    + ("lambda L_lambda: %{y:.4e} Lsun" if y_col == "lambda_l_lambda" else "F_lambda: %{y:.4e}")
                    + "<extra></extra>"
                ),
            ))

    if bandpass_v2 and not point_model_df.empty:
        point_modes = ["intrinsic"] if mode in corrected_modes else ["observed"]
        if mode == "both":
            point_modes = ["observed", "intrinsic"]
        for point_mode in point_modes:
            if point_mode == "intrinsic" and not intrinsic_ratio_complete:
                continue
            if y_col == "lambda_l_lambda":
                point_y_col = "model_lambda_l_lambda_intrinsic" if point_mode == "intrinsic" else "model_lambda_l_lambda"
                point_y_scale = LSUN_ERG_S
            else:
                point_y_col = "model_flux_lambda_intrinsic" if point_mode == "intrinsic" else "model_flux_lambda"
                point_y_scale = 1.0
            if point_y_col not in point_model_df.columns:
                continue
            predicted = point_model_df.copy()
            predicted_plot_x = pd.to_numeric(
                predicted.get(
                    "plot_lambda_angstrom",
                    pd.Series(np.nan, index=predicted.index, dtype=float),
                ),
                errors="coerce",
            )
            predicted_legacy_x = pd.to_numeric(
                predicted.get(
                    "lambda_eff_angstrom",
                    pd.Series(np.nan, index=predicted.index, dtype=float),
                ),
                errors="coerce",
            )
            predicted["x"] = predicted_plot_x.where(
                np.isfinite(predicted_plot_x) & (predicted_plot_x > 0),
                predicted_legacy_x,
            )
            predicted["y"] = pd.to_numeric(predicted.get(point_y_col), errors="coerce") / point_y_scale
            predicted = predicted[np.isfinite(predicted["x"]) & np.isfinite(predicted["y"]) & (predicted["x"] > 0) & (predicted["y"] > 0)]
            if not predicted.empty:
                predicted["used_bool"] = predicted.get(
                    "used", pd.Series(0, index=predicted.index, dtype=int)
                ).map(_to_bool_int).astype(bool)
                for used, group in predicted.groupby("used_bool"):
                    mode_suffix = f" ({point_mode})" if mode == "both" else ""
                    fig.add_trace(go.Scatter(
                        x=group["x"],
                        y=group["y"],
                        mode="markers",
                        name=("CK synthetic fitted bands" if used else "CK synthetic diagnostics") + mode_suffix,
                        marker=dict(
                            size=9,
                            symbol="x" if used else "circle-open",
                            color=model_color if used else spec["muted"],
                            line=dict(width=1.2, color=model_color if used else spec["muted"]),
                        ),
                        customdata=np.column_stack([
                            group.get("source", "").astype(str),
                            group.get("band", "").astype(str),
                            group.get("exclusion_reason", "").fillna("").astype(str),
                            group.get("prediction_reason", "").fillna("").astype(str),
                            pd.to_numeric(group.get("residual_sigma"), errors="coerce"),
                        ]),
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>"
                            "source: %{customdata[0]} %{customdata[1]}<br>"
                            "lambda: %{x:.5g} A<br>"
                            "fit reason: %{customdata[2]}<br>"
                            "prediction reason: %{customdata[3]}<br>"
                            "residual: %{customdata[4]:.3g} sigma<extra></extra>"
                        ),
                    ))

    if fig.data:
        model_traces = [
            trace for trace in fig.data
            if "Castelli/Kurucz" in str(getattr(trace, "name", "") or "")
        ]
        if model_traces:
            marker_traces = [
                trace for trace in fig.data
                if "Castelli/Kurucz" not in str(getattr(trace, "name", "") or "")
            ]
            fig.data = tuple([*model_traces, *marker_traces])

    x_title = r"$\lambda\ [\mathring{\mathrm{A}}]$"
    y_title = (
        r"$\lambda L_{\lambda}\ [L_{\odot}]$"
        if y_col == "lambda_l_lambda"
        else r"$F_{\lambda}\ [\mathrm{erg\,s^{-1}\,cm^{-2}}\,\mathring{\mathrm{A}}^{-1}]$"
    )
    fig.update_layout(
        title="Spectral Energy Distribution",
        height=390,
        margin=dict(l=72, r=18, t=82, b=62),
        paper_bgcolor=spec["paper"],
        plot_bgcolor=spec["plot"],
        font=dict(color=spec["font"], family=PUBLICATION_PLOTLY_FONT, size=10),
        legend=dict(orientation="h", y=1.17, x=0.0, bgcolor="rgba(0,0,0,0)"),
    )
    x_range = _log_axis_range_from_data(phot_x, pad_dex=0.10, min_span_dex=0.45)
    y_range = _log_axis_range_from_data(phot_y, pad_dex=0.16, min_span_dex=0.60)
    fig.update_xaxes(title=x_title, type="log", gridcolor=spec["grid"], zeroline=False, range=x_range)
    fig.update_yaxes(title=y_title, type="log", gridcolor=spec["grid"], zeroline=False, range=y_range)
    return fig, sed_df, warnings


def _linear_log_axis_limits(
    values: Iterable[object],
    *,
    pad_dex: float = 0.08,
    min_span_dex: float = 0.35,
) -> tuple[float, float] | None:
    """Convert review SED log-axis ranges to linear matplotlib limits."""

    log_range = _log_axis_range_from_data(
        values,
        pad_dex=pad_dex,
        min_span_dex=min_span_dex,
    )
    if log_range is None:
        return None
    return float(10.0 ** log_range[0]), float(10.0 ** log_range[1])


def _sed_row_matplotlib_marker(row: pd.Series) -> tuple[str, dict[str, object]]:
    flags = str(row.get("quality_flags") or "")
    if _to_bool_int(row.get("is_upper_limit", 0)):
        return "v", {}
    if _to_bool_int(row.get("is_synthetic", 0)):
        return "o", {"facecolors": "none", "edgecolors": None, "linewidths": 0.9}
    if "confusion_risk" in flags:
        return "D", {"facecolors": "none", "edgecolors": None, "linewidths": 0.9}
    return "o", {}


def _draw_sed_photometry_matplotlib(
    ax,
    sed_df: pd.DataFrame,
    *,
    y_col: str,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    plot_df = sed_df.copy()
    plot_x = pd.to_numeric(plot_df["plot_lambda_angstrom"], errors="coerce")
    legacy_x = pd.to_numeric(plot_df["lambda_eff_angstrom"], errors="coerce")
    plot_df["x"] = plot_x.where(np.isfinite(plot_x) & (plot_x > 0), legacy_x)
    y_scale = LSUN_ERG_S if y_col == "lambda_l_lambda" else 1.0
    plot_df["y"] = pd.to_numeric(plot_df[y_col], errors="coerce") / y_scale
    plot_df = plot_df[
        np.isfinite(plot_df["x"])
        & np.isfinite(plot_df["y"])
        & (plot_df["x"] > 0)
        & (plot_df["y"] > 0)
    ]
    phot_x = plot_df["x"].to_numpy(dtype=float)
    phot_y = plot_df["y"].to_numpy(dtype=float)
    y_err_col = "lambda_l_lambda_err" if y_col == "lambda_l_lambda" else "flux_lambda_err"

    source_groups = list(
        plot_df.groupby(["sed_mode", "source"], dropna=False, sort=False)
    )
    # The legend uses the first labelled artist from each source.  Order those
    # artists by their first plotted wavelength, not alphabetically, so the
    # legend follows the SED from left to right.
    source_groups.sort(
        key=lambda item: (
            float(np.nanmin(item[1]["x"].to_numpy(dtype=float))),
            str(item[0][0]),
            str(item[0][1]),
        )
    )
    for (mode_name, source), grp in source_groups:
        color = SOURCE_COLORS.get(str(source), "#bbbbbb")
        label = f"{source} ({mode_name})" if mode == "both" else str(source)
        is_sampled_spectrum = bool(
            len(grp) > 1
            and grp["quality_flags"]
            .fillna("")
            .astype(str)
            .str.contains("correlated_spectrum", regex=False)
            .all()
        )
        x = grp["x"].to_numpy(dtype=float)
        y = grp["y"].to_numpy(dtype=float)
        if is_sampled_spectrum:
            order = np.argsort(x)
            ax.plot(
                x[order],
                y[order],
                color=color,
                linewidth=1.2,
                alpha=1.0,
                label=label,
                zorder=2,
                rasterized=True,
            )
            continue

        y_err = None
        if y_err_col in grp.columns:
            err_vals = pd.to_numeric(grp[y_err_col], errors="coerce") / y_scale
            if np.isfinite(err_vals).any():
                y_err = err_vals.to_numpy(dtype=float)

        if y_err is not None and len(grp) == len(y_err):
            ax.errorbar(
                x,
                y,
                yerr=np.where(np.isfinite(y_err), y_err, np.nan),
                fmt="none",
                ecolor=color,
                elinewidth=0.6,
                capsize=1.0,
                alpha=0.9,
                zorder=1,
                rasterized=True,
            )

        first_row = True
        for _, row in grp.iterrows():
            marker, marker_kwargs = _sed_row_matplotlib_marker(row)
            scatter_kwargs: dict[str, object] = {
                "s": 18,
                "marker": marker,
                "alpha": 0.9,
                "zorder": 2,
                "rasterized": True,
                "label": label if first_row else "_nolegend_",
            }
            if marker_kwargs.get("facecolors") == "none":
                scatter_kwargs["facecolors"] = "none"
                scatter_kwargs["edgecolors"] = color
                scatter_kwargs["linewidths"] = marker_kwargs.get("linewidths", 0.9)
            else:
                scatter_kwargs["color"] = color
            ax.scatter(row["x"], row["y"], **scatter_kwargs)
            first_row = False

    return phot_x, phot_y


def _draw_sed_model_matplotlib(
    ax,
    *,
    model_curve_rows: pd.DataFrame | Iterable[dict] | None,
    model_fit_rows: pd.DataFrame | Iterable[dict] | None,
    y_col: str,
    mode: str,
    intrinsic_ratio_complete: bool,
    theme: str | None,
) -> None:
    corrected_modes = {"corrected", "ism-corrected", "ism_corrected", "dereddened"}
    model_df = pd.DataFrame(model_curve_rows) if model_curve_rows is not None else pd.DataFrame()
    fit_df = pd.DataFrame(model_fit_rows) if model_fit_rows is not None else pd.DataFrame()
    fit_version = str(fit_df.iloc[0].get("fit_version") or "") if not fit_df.empty else ""
    bandpass_v2 = fit_version.startswith("ck04-bandpass-v")
    fit_status = str(fit_df.iloc[0].get("status") or "").strip().lower() if not fit_df.empty else ""
    legacy_curve = not bandpass_v2 and fit_status == "ok"
    model_is_comparable = legacy_curve or (
        bandpass_v2 and (mode not in corrected_modes or intrinsic_ratio_complete)
    )
    if not model_is_comparable or model_df.empty or "wavelength_angstrom" not in model_df.columns:
        return

    model_color = "#111827" if str(theme or "black").strip().lower() == "white" else "#f8fafc"
    curve_modes = ["legacy"] if legacy_curve else (
        ["observed"] if mode not in corrected_modes and mode != "both" else ["intrinsic"]
    )
    if mode == "both" and not legacy_curve:
        curve_modes = ["observed", "intrinsic"]
    y_scale = LSUN_ERG_S if y_col == "lambda_l_lambda" else 1.0

    for curve_mode in curve_modes:
        if curve_mode == "intrinsic" and not intrinsic_ratio_complete:
            continue
        if bandpass_v2:
            curve_col = (
                f"lambda_l_lambda_{curve_mode}"
                if y_col == "lambda_l_lambda"
                else f"flux_lambda_{curve_mode}"
            )
        else:
            curve_col = y_col
        if curve_col not in model_df.columns:
            continue
        curve = model_df.copy()
        curve["x"] = pd.to_numeric(curve["wavelength_angstrom"], errors="coerce")
        curve["y"] = pd.to_numeric(curve[curve_col], errors="coerce") / y_scale
        curve = curve[
            np.isfinite(curve["x"])
            & np.isfinite(curve["y"])
            & (curve["x"] > 0)
            & (curve["y"] > 0)
        ]
        if curve.empty:
            continue
        curve = curve.sort_values("x")
        teff = _safe_float(curve.iloc[0].get("teff_k"))
        name = (
            "Castelli/Kurucz fit"
            if curve_mode == "legacy"
            else f"Castelli/Kurucz {curve_mode} fit"
        )
        if teff is not None:
            name += f" ({teff:.0f} K)"
        ax.plot(
            curve["x"],
            curve["y"],
            color=model_color,
            linewidth=1.4,
            alpha=0.95,
            linestyle="--" if curve_mode == "intrinsic" and mode == "both" else "-",
            label=name,
            zorder=3,
        )


def render_sed_matplotlib(
    ax,
    payload: dict,
    *,
    candidate_id: str | None = None,
    external_rows: pd.DataFrame | Iterable[dict] | None = None,
    model_curve_rows: pd.DataFrame | Iterable[dict] | None = None,
    model_fit_rows: pd.DataFrame | Iterable[dict] | None = None,
    model_point_rows: pd.DataFrame | Iterable[dict] | None = None,
    extinction_mode: str = "observed",
    theme: str | None = "white",
    y_axis_side: str = "right",
) -> list[str]:
    """Render the browser-review SED on a matplotlib axis."""

    mode = str(extinction_mode or "observed").strip().lower()
    _fig, sed_df, warnings = build_sed_figure(
        payload,
        candidate_id=candidate_id,
        external_rows=external_rows,
        model_curve_rows=model_curve_rows,
        model_fit_rows=model_fit_rows,
        model_point_rows=model_point_rows,
        extinction_mode=extinction_mode,
        theme=theme,
    )

    if sed_df is None or sed_df.empty:
        spec = _theme(theme)
        ax.set_facecolor(spec["plot"])
        ax.text(
            0.5,
            0.5,
            "No SED photometry available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=spec["muted"],
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return warnings

    y_col = "lambda_l_lambda"
    if sed_df[y_col].isna().all():
        y_col = "flux_lambda"

    phot_x, phot_y = _draw_sed_photometry_matplotlib(
        ax,
        sed_df,
        y_col=y_col,
        mode=mode,
    )
    corrected_modes = {"corrected", "ism-corrected", "ism_corrected", "dereddened"}
    intrinsic_ratio_complete = False
    if mode in corrected_modes or mode == "both":
        point_model_df = (
            pd.DataFrame(model_point_rows) if model_point_rows is not None else pd.DataFrame()
        )
        if not point_model_df.empty and not sed_df.empty:
            corrected_expected = int((sed_df["sed_mode"].astype(str).str.lower() != "observed").sum())
            corrected_applied = 0
            for _, row in sed_df.iterrows():
                if str(row.get("sed_mode") or "").lower() == "observed":
                    continue
                plot_lambda = _safe_float(row.get("plot_lambda_angstrom")) or _safe_float(
                    row.get("lambda_eff_angstrom")
                )
                identity = _fit_normalization_identity(row)
                if not all(identity) or plot_lambda is None:
                    continue
                match = point_model_df[
                    point_model_df.apply(
                        lambda point: _fit_normalization_identity(point) == identity,
                        axis=1,
                    )
                ]
                if len(match) != 1:
                    continue
                point = match.iloc[0]
                model_observed_fnu = _safe_float(point.get("model_flux_nu_jy"))
                model_intrinsic_fnu = _safe_float(point.get("model_flux_nu_jy_intrinsic"))
                if (
                    model_observed_fnu is not None
                    and model_observed_fnu > 0
                    and model_intrinsic_fnu is not None
                    and model_intrinsic_fnu > 0
                ):
                    corrected_applied += 1
            intrinsic_ratio_complete = corrected_expected > 0 and corrected_applied == corrected_expected

    _draw_sed_model_matplotlib(
        ax,
        model_curve_rows=model_curve_rows,
        model_fit_rows=model_fit_rows,
        y_col=y_col,
        mode=mode,
        intrinsic_ratio_complete=intrinsic_ratio_complete,
        theme=theme,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    x_limits = _linear_log_axis_limits(phot_x, pad_dex=0.10, min_span_dex=0.45)
    y_limits = _linear_log_axis_limits(phot_y, pad_dex=0.16, min_span_dex=0.60)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)

    ax.set_xlabel(r"$\lambda\,[\AA]$")
    if y_col == "lambda_l_lambda":
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

        def plain_luminosity_tick(value: float, _position: int) -> str:
            if not np.isfinite(value) or value <= 0:
                return ""
            return np.format_float_positional(
                float(value),
                precision=6,
                unique=True,
                fractional=False,
                trim="-",
            )

        plain_formatter = FuncFormatter(plain_luminosity_tick)
        ax.yaxis.set_major_formatter(plain_formatter)
        if y_limits is not None and math.log10(y_limits[1] / y_limits[0]) <= 1.5:
            ax.yaxis.set_minor_locator(
                LogLocator(base=10.0, subs=(2.0, 3.0, 4.0, 6.0))
            )
            ax.yaxis.set_minor_formatter(plain_formatter)
        else:
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel(r"$\lambda L_\lambda/L_\odot$")
    else:
        ax.set_ylabel(r"$F_\lambda$")
    if str(y_axis_side or "right").strip().lower() == "left":
        ax.yaxis.set_label_position("left")
        ax.yaxis.tick_left()
        ax.yaxis.set_label_coords(-0.16, 0.5)
    else:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.yaxis.set_label_coords(1.04, 0.5)
    ax.tick_params(length=3.0, width=0.7, pad=1.5)

    handles, labels = ax.get_legend_handles_labels()
    legend_entries = [
        (handle, label)
        for handle, label in zip(handles, labels)
        if not str(label).startswith("Castelli/Kurucz")
    ]
    if legend_entries:
        handles, labels = zip(*legend_entries)
        ax.legend(
            handles[:6],
            labels[:6],
            loc="lower left",
            frameon=True,
            framealpha=0.85,
            handlelength=1.0,
            borderpad=0.25,
        )
    return warnings


def _current_fit_normalization_references(
    conn: sqlite3.Connection,
    candidate_id: str,
) -> pd.DataFrame:
    """Return the exact normalization triples referenced by the current fit.

    The immutable fit-input ledger is authoritative.  When the legacy/current
    model-point snapshot is also present, both views must agree exactly; an
    inconsistent snapshot deliberately yields no references so the review path
    falls back to the baseline normalization and flags the fit as stale.
    """

    columns = ["measurement_id", "normalization_version", "normalization_hash"]
    try:
        fit = conn.execute(
            "SELECT fit_run_id, fit_run_hash FROM sed_model_fits WHERE candidate_id = ?",
            (str(candidate_id),),
        ).fetchone()
    except Exception:
        return pd.DataFrame(columns=columns)
    if fit is None:
        return pd.DataFrame(columns=columns)
    fit_run_id = _clean_text(fit[0])
    fit_run_hash = _clean_text(fit[1])

    ledger = pd.DataFrame(columns=columns)
    if fit_run_id:
        try:
            ledger = pd.read_sql_query(
                "SELECT measurement_id, normalization_version, normalization_hash "
                "FROM sed_fit_inputs WHERE fit_run_id = ?",
                conn,
                params=(fit_run_id,),
            )
        except Exception:
            ledger = pd.DataFrame(columns=columns)

    try:
        if fit_run_id:
            point_sql = (
                "SELECT measurement_id, normalization_version, normalization_hash "
                "FROM sed_model_points WHERE candidate_id = ? AND fit_run_id = ?"
            )
            point_params: tuple[object, ...] = (str(candidate_id), fit_run_id)
        elif fit_run_hash:
            point_sql = (
                "SELECT measurement_id, normalization_version, normalization_hash "
                "FROM sed_model_points WHERE candidate_id = ? AND fit_run_hash = ?"
            )
            point_params = (str(candidate_id), fit_run_hash)
        else:
            point_sql = (
                "SELECT measurement_id, normalization_version, normalization_hash "
                "FROM sed_model_points WHERE candidate_id = ?"
            )
            point_params = (str(candidate_id),)
        points = pd.read_sql_query(point_sql, conn, params=point_params)
    except Exception:
        points = pd.DataFrame(columns=columns)

    def exact(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=columns)
        out = frame[columns].copy()
        for column in columns:
            out[column] = out[column].map(_clean_text)
        out = out[(out[columns] != "").all(axis=1)].drop_duplicates()
        counts = out.groupby("measurement_id", dropna=False).size()
        valid_ids = set(counts[counts == 1].index.astype(str))
        return out[out["measurement_id"].isin(valid_ids)].reset_index(drop=True)

    ledger = exact(ledger)
    points = exact(points)
    if not ledger.empty and not points.empty:
        ledger_set = set(map(tuple, ledger[columns].itertuples(index=False, name=None)))
        point_set = set(map(tuple, points[columns].itertuples(index=False, name=None)))
        if ledger_set != point_set:
            return pd.DataFrame(columns=columns)
        return ledger
    return ledger if not ledger.empty else points


def _load_review_sed_normalizations(
    conn: sqlite3.Connection,
    candidate_id: str,
) -> pd.DataFrame:
    """Load exact current-fit normalizations plus baseline rows for new inputs."""

    from malca.review.sed_storage import load_prepared_sed_measurements

    baseline = load_prepared_sed_measurements(
        conn,
        str(candidate_id),
        normalization_version=CANONICAL_SED_NORMALIZATION_VERSION,
    )
    legacy_baseline = load_prepared_sed_measurements(
        conn,
        str(candidate_id),
        normalization_version=LEGACY_CANONICAL_SED_NORMALIZATION_VERSION,
    )
    if not legacy_baseline.empty:
        current_ids = set(
            baseline.get("measurement_id", pd.Series(dtype=str)).map(_clean_text)
        )
        if current_ids:
            legacy_baseline = legacy_baseline.loc[
                ~legacy_baseline["measurement_id"].map(_clean_text).isin(current_ids)
            ]
        baseline = _concat_sed_frames(baseline, legacy_baseline)
    references = _current_fit_normalization_references(conn, str(candidate_id))
    if references.empty:
        return baseline

    exact_parts: list[pd.DataFrame] = []
    for version, expected in references.groupby("normalization_version", sort=False):
        loaded = load_prepared_sed_measurements(
            conn,
            str(candidate_id),
            normalization_version=str(version),
        )
        if loaded.empty:
            continue
        expected_identity = {
            (
                _clean_text(row.get("measurement_id")),
                _clean_text(row.get("normalization_version")),
                _clean_text(row.get("normalization_hash")),
            )
            for _, row in expected.iterrows()
        }
        keep = loaded.apply(
            lambda row: (
                _clean_text(row.get("measurement_id")),
                _clean_text(row.get("normalized_normalization_version")),
                _clean_text(row.get("normalized_normalization_hash")),
            )
            in expected_identity,
            axis=1,
        )
        exact_parts.append(loaded.loc[keep].copy())

    exact = (
        _concat_sed_frames(*exact_parts)
        if exact_parts
        else pd.DataFrame(columns=baseline.columns)
    )
    exact_ids = set(exact.get("measurement_id", pd.Series(dtype=str)).map(_clean_text))
    if not baseline.empty and exact_ids:
        baseline = baseline.loc[
            ~baseline["measurement_id"].map(_clean_text).isin(exact_ids)
        ].copy()
    if exact.empty:
        return baseline
    if baseline.empty:
        return exact.reset_index(drop=True)
    return _concat_sed_frames(exact, baseline).reset_index(drop=True)


def load_sed_rows(conn: sqlite3.Connection, candidate_id: str) -> pd.DataFrame:
    try:
        prepared = _load_review_sed_normalizations(conn, str(candidate_id))
    except Exception:
        prepared = pd.DataFrame()
    if not prepared.empty:
        records: list[dict[str, object]] = []
        for _, item in prepared.iterrows():
            observable = str(item.get("observable_kind") or "").strip().lower()
            if observable == "quoted_fnu":
                mag = None
                mag_err = None
                mag_system = "Jy"
            else:
                mag = _safe_float(item.get("native_value"))
                mag_err = _safe_float(item.get("native_error"))
                mag_system = "AB" if observable == "ab_mag" else "Vega" if observable == "vega_mag" else ""
            plot_lambda = _safe_float(item.get("normalized_plot_lambda_angstrom"))
            flux_nu = _safe_float(item.get("normalized_flux_nu_jy"))
            records.append(
                {
                    "candidate_id": str(item.get("candidate_id") or candidate_id),
                    "source": str(item.get("source") or ""),
                    "band": str(item.get("band") or ""),
                    "mag": mag,
                    "mag_err": mag_err,
                    "mag_system": mag_system,
                    "lambda_eff_angstrom": plot_lambda,
                    "flux_lambda": _safe_float(item.get("normalized_flux_lambda")),
                    "flux_lambda_err": _safe_float(item.get("normalized_flux_lambda_err")),
                    "lambda_l_lambda": _safe_float(item.get("normalized_lambda_l_lambda")),
                    "lambda_l_lambda_err": _safe_float(item.get("normalized_lambda_l_lambda_err")),
                    "flux_nu_jy": flux_nu,
                    "flux_nu_jy_err": _safe_float(item.get("normalized_flux_nu_jy_err")),
                    "sep_arcsec": _safe_float(item.get("match_sep_arcsec")),
                    "is_synthetic": _to_bool_int(item.get("is_synthetic")),
                    "is_upper_limit": _to_bool_int(item.get("is_upper_limit")),
                    "quality_flags": str(item.get("quality_flags") or ""),
                    "svo_filter_id": str(item.get("response_id") or "") or None,
                    "av_coeff": None,
                    "measurement_id": str(item.get("measurement_id") or ""),
                    "normalization_version": str(item.get("normalized_normalization_version") or ""),
                    "catalog_release": item.get("release"),
                    "source_object_id": item.get("catalog_object_id"),
                    "catalog_measurement_id": item.get("catalog_measurement_id"),
                    "instrument": item.get("instrument"),
                    "exposure_id": item.get("exposure_id"),
                    "epoch_mjd": _safe_float(item.get("epoch_mjd")),
                    "correlation_group": item.get("correlation_group"),
                    "provenance_json": item.get("provenance_json"),
                    "native_value": _safe_float(item.get("native_value")),
                    "native_error": _safe_float(item.get("native_error")),
                    "native_unit": item.get("native_unit"),
                    "observable_kind": observable,
                    "passband_fidelity": item.get("passband_fidelity"),
                    "observed_flux_nu_jy": flux_nu,
                    "observed_flux_nu_jy_err": _safe_float(item.get("normalized_flux_nu_jy_err")),
                    "plot_lambda_angstrom": plot_lambda,
                    "plot_lambda_kind": item.get("normalized_plot_lambda_kind"),
                    "lambda_nominal_angstrom": _safe_float(item.get("normalized_lambda_nominal_angstrom")),
                    "lambda_pivot_angstrom": _safe_float(item.get("normalized_lambda_pivot_angstrom")),
                    "lambda_reference_angstrom": _safe_float(item.get("normalized_lambda_reference_angstrom")),
                    "lambda_isophotal_angstrom": _safe_float(item.get("normalized_lambda_isophotal_angstrom")),
                    "response_kind": item.get("passband_fidelity"),
                    "fit_policy": item.get("fit_policy"),
                    "native_flux_unit": item.get("native_unit"),
                    "calibration_source": item.get("calibration_id"),
                    "calibration_id": item.get("calibration_id"),
                    "calibration_hash": item.get("normalized_calibration_hash"),
                    "response_hash": item.get("normalized_response_hash"),
                    "normalization_hash": item.get("normalized_normalization_hash"),
                }
            )
        return pd.DataFrame(records, columns=CANONICAL_SED_COLUMNS)
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
    canonical_frame = rows.copy()
    frame = canonical_frame.copy()
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
    savepoint = "sed_rows_v3"
    conn.execute(f"SAVEPOINT {savepoint}")
    try:
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
        has_v3_storage = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sed_measurements'"
        ).fetchone()
        if has_v3_storage is not None:
            from malca.review.sed_storage import store_canonical_sed_rows

            store_canonical_sed_rows(conn, canonical_frame, commit=False)
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
    except Exception:
        conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        raise
    conn.commit()
    return count


def _candidate_id_for_row(row: pd.Series) -> str:
    for col in ("candidate_id", "asas_sn_id", "gaia_id", "source_id"):
        if col in row and str(row.get(col) or "").strip():
            return str(row.get(col)).strip()
    return str(row.name)


ProgressCallback = Callable[[str], None]


def _sed_cache_path(source_key: str) -> Path:
    token = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in source_key.lower()).strip("_")
    return SED_CACHE_DIR / f"{token or 'source'}.parquet"


def _source_fetch_signature(source_key: str) -> tuple[SedFetchSignature, str]:
    policy = SED_SOURCE_FETCH_SIGNATURES.get(
        str(source_key).strip().lower(),
        SedFetchSignature(
            catalog_release=str(source_key).strip().lower(),
            adapter_version="legacy-adapter-v1",
            match_policy_version="nearest-neighbor-v1",
            coordinate_epoch="catalog-default",
            quality_policy_version="legacy-quality-v1",
        ),
    )
    payload = {
        "source_key": str(source_key).strip().lower(),
        "catalog_release": policy.catalog_release,
        "adapter_version": policy.adapter_version,
        "match_policy_version": policy.match_policy_version,
        "coordinate_epoch": policy.coordinate_epoch,
        "quality_policy_version": policy.quality_policy_version,
    }
    signature = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return policy, signature


def _candidate_astrometry_hash(row: pd.Series) -> str:
    def first(*names: str) -> object | None:
        for name in names:
            if name in row:
                value = row.get(name)
                try:
                    if value is not None and not pd.isna(value):
                        return value
                except (TypeError, ValueError):
                    if value is not None:
                        return value
        return None

    payload = {
        "candidate_id": _candidate_id_for_row(row),
        "ra_deg": _safe_float(first("ra_deg", "ra", "RA")),
        "dec_deg": _safe_float(first("dec_deg", "dec", "DEC")),
        "pmra_masyr": _safe_float(first("pmra", "gaia_pmra")),
        "pmdec_masyr": _safe_float(first("pmdec", "gaia_pmdec")),
        "ref_epoch_jyear": _safe_float(first("ref_epoch", "gaia_ref_epoch")) or 2016.0,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _expected_astrometry_hashes(df: pd.DataFrame) -> dict[str, str]:
    return {
        _candidate_id_for_row(row): _candidate_astrometry_hash(row)
        for _, row in pd.DataFrame(df).iterrows()
    }


def _cache_signature_mask(
    cache: pd.DataFrame,
    source_key: str,
    astrometry_hashes: Mapping[str, str],
) -> pd.Series:
    if cache.empty:
        return pd.Series(False, index=cache.index, dtype=bool)
    _policy, expected_signature = _source_fetch_signature(source_key)
    if "_cache_fetch_signature" not in cache.columns or "_cache_astrometry_hash" not in cache.columns:
        return pd.Series(False, index=cache.index, dtype=bool)
    expected_astrometry = cache["_cache_candidate_id"].astype(str).map(astrometry_hashes)
    return (
        cache["_cache_fetch_signature"].fillna("").astype(str).eq(expected_signature)
        & cache["_cache_astrometry_hash"].fillna("").astype(str).eq(
            expected_astrometry.fillna("").astype(str)
        )
    )


def _sed_candidate_ids(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=str)
    return df.apply(_candidate_id_for_row, axis=1).astype(str)


def _canonical_sed_json_text(value: object) -> str | None:
    """Return deterministic JSON text for Parquet/SQLite JSON columns."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _normalize_sed_json_text_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize legacy struct-valued JSON fields to scalar JSON strings."""
    if frame.empty or "provenance_json" not in frame.columns:
        return frame
    provenance = frame["provenance_json"]
    present = provenance.dropna()
    if present.empty or bool(present.map(lambda value: isinstance(value, str)).all()):
        return frame
    # Avoid a deep copy of multi-million-row SED frames when repairing one
    # legacy object column.  Assigning on a shallow frame replaces only that
    # column while leaving the caller's frame untouched.
    out = frame.copy(deep=False)
    out["provenance_json"] = provenance.map(_canonical_sed_json_text)
    return out


def _read_sed_source_cache(source_key: str) -> pd.DataFrame:
    cache, error = _read_sed_source_cache_with_error(source_key)
    if error:
        print(f"[SED] cache warning: {error}")
    return cache


def _read_sed_source_cache_with_error(source_key: str) -> tuple[pd.DataFrame, str | None]:
    path = _sed_cache_path(source_key).expanduser()
    if not path.exists():
        return pd.DataFrame(), None
    try:
        cache = pd.read_parquet(path)
    except Exception as exc:
        return pd.DataFrame(), f"could not read {path}: {exc}"
    if "_cache_candidate_id" not in cache.columns:
        return pd.DataFrame(), f"{path} has no _cache_candidate_id column"
    cache = _normalize_sed_json_text_columns(cache)
    cache["_cache_candidate_id"] = cache["_cache_candidate_id"].astype(str)
    return cache, None


def _sed_measurement_dedupe_key(
    frame: pd.DataFrame,
    *,
    candidate_column: str = "candidate_id",
) -> pd.Series:
    """Return identities that preserve multi-epoch and multi-instrument rows."""
    if frame.empty:
        return pd.Series(dtype=str, index=frame.index)
    measurement_ids = frame.get(
        "measurement_id", pd.Series("", index=frame.index, dtype=object)
    ).map(_clean_text)
    identity_columns = (
        candidate_column,
        "source",
        "band",
        "catalog_release",
        "source_object_id",
        "catalog_measurement_id",
        "instrument",
        "exposure_id",
        "epoch_mjd",
        "quality_flags",
    )
    fallback = pd.Series("legacy", index=frame.index, dtype=object)
    for column in identity_columns:
        values = frame.get(column, pd.Series("", index=frame.index, dtype=object)).map(_clean_text)
        fallback = fallback + "|" + values
    return measurement_ids.where(measurement_ids.ne(""), fallback)


def _write_sed_source_cache(source_key: str, rows: pd.DataFrame) -> bool:
    if rows.empty:
        return True
    path = _sed_cache_path(source_key).expanduser()
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        try:
            existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        except Exception as exc:
            print(f"[SED] cache warning: replacing unreadable {path}: {exc}")
            existing = pd.DataFrame()
        if not existing.empty and "_cache_candidate_id" not in existing.columns:
            print(f"[SED] cache warning: replacing legacy cache without candidate IDs: {path}")
            existing = pd.DataFrame()
        if not existing.empty and "_cache_candidate_id" in rows.columns:
            refreshed_ids = set(rows["_cache_candidate_id"].astype(str))
            existing = existing.loc[~existing["_cache_candidate_id"].astype(str).isin(refreshed_ids)].copy()
        combined = pd.concat([existing, rows], ignore_index=True) if not existing.empty else rows.copy()
        combined = _normalize_sed_json_text_columns(combined)
        if "_cache_candidate_id" in combined.columns:
            combined["_cache_row_identity"] = _sed_measurement_dedupe_key(
                combined,
                candidate_column="_cache_candidate_id",
            )
            combined = combined.drop_duplicates(subset=["_cache_row_identity"], keep="last")
            combined = combined.drop(columns=["_cache_row_identity"])
        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(
            temporary_path,
            index=False,
            compression=PARQUET_CACHE_COMPRESSION,
        )
        os.replace(temporary_path, path)
        return True
    except Exception as exc:
        print(f"[SED] cache warning: could not write {path}: {exc}")
        try:
            temporary_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def _set_fetch_statuses(frame: pd.DataFrame, statuses: Mapping[str, str]) -> pd.DataFrame:
    frame.attrs[SED_FETCH_STATUS_ATTR] = {str(key): str(value) for key, value in statuses.items()}
    return frame


def _fetch_statuses(frame: pd.DataFrame | None) -> dict[str, str]:
    if frame is None:
        return {}
    raw = frame.attrs.get(SED_FETCH_STATUS_ATTR, {})
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): str(value).strip().lower() for key, value in raw.items()}


def _cache_rows_for_sed_result(
    source_key: str,
    input_ids: set[str],
    fetched: pd.DataFrame,
    *,
    status_by_candidate: Mapping[str, str] | None = None,
    astrometry_hashes: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    updated_at = pd.Timestamp.utcnow().isoformat()
    policy, fetch_signature = _source_fetch_signature(source_key)
    astrometry_map = {
        str(key): str(value)
        for key, value in (astrometry_hashes or {}).items()
    }
    frames: list[pd.DataFrame] = []
    hit_ids: set[str] = set()
    status_map = {
        str(key): str(value).strip().lower()
        for key, value in (status_by_candidate or {}).items()
    }
    if fetched is not None and not fetched.empty:
        hit_rows = fetched.copy()
        for col in CANONICAL_SED_COLUMNS:
            if col not in hit_rows.columns:
                hit_rows[col] = None
        hit_rows = hit_rows[CANONICAL_SED_COLUMNS]
        hit_rows.insert(0, "_cache_candidate_id", hit_rows["candidate_id"].astype(str))
        hit_rows["_cache_status"] = hit_rows["_cache_candidate_id"].map(
            lambda cid: (
                "partial"
                if status_map.get(str(cid), "hit") in SED_CACHE_RETRYABLE_STATUSES
                else status_map.get(str(cid), "hit")
                if status_map.get(str(cid), "hit") in SED_CACHE_TERMINAL_STATUSES
                else "hit"
            )
        )
        hit_rows["_cache_updated_at"] = updated_at
        hit_ids = set(hit_rows["_cache_candidate_id"].astype(str))
        frames.append(hit_rows)

    non_hit_ids = sorted(input_ids - hit_ids)
    for status in (*sorted(SED_CACHE_TERMINAL_STATUSES), *sorted(SED_CACHE_RETRYABLE_STATUSES)):
        if status == "hit":
            continue
        status_ids = [cid for cid in non_hit_ids if status_map.get(cid, "error") == status]
        if not status_ids:
            continue
        status_rows = pd.DataFrame([{col: None for col in CANONICAL_SED_COLUMNS} for _ in status_ids])
        status_rows["candidate_id"] = status_ids
        status_rows["source"] = source_key
        status_rows.insert(0, "_cache_candidate_id", status_ids)
        status_rows["_cache_status"] = status
        status_rows["_cache_updated_at"] = updated_at
        frames.append(status_rows)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["_cache_catalog_release"] = policy.catalog_release
    combined["_cache_adapter_version"] = policy.adapter_version
    combined["_cache_match_policy_version"] = policy.match_policy_version
    combined["_cache_coordinate_epoch"] = policy.coordinate_epoch
    combined["_cache_quality_policy_version"] = policy.quality_policy_version
    combined["_cache_fetch_signature"] = fetch_signature
    combined["_cache_astrometry_hash"] = combined["_cache_candidate_id"].astype(str).map(
        astrometry_map
    )
    return combined


def _merge_sed_measurement_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    usable = [
        pd.DataFrame(frame).reindex(columns=CANONICAL_SED_COLUMNS)
        for frame in frames
        if frame is not None and not frame.empty
    ]
    if not usable:
        return pd.DataFrame(columns=CANONICAL_SED_COLUMNS)
    combined = pd.concat(usable, ignore_index=True)
    combined["_measurement_key"] = _sed_measurement_dedupe_key(combined)
    return (
        combined.drop_duplicates(subset=["_measurement_key"], keep="last")
        .drop(columns=["_measurement_key"])
        .reset_index(drop=True)
    )


def _candidate_fetch_outcomes(
    candidate_ids: set[str],
    fetched: pd.DataFrame,
    explicit_statuses: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve per-candidate outcomes without turning unknown failures into misses."""
    statuses = {
        str(key): str(value).strip().lower()
        for key, value in (explicit_statuses or {}).items()
    }
    row_ids = (
        set(pd.DataFrame(fetched)["candidate_id"].dropna().astype(str))
        if fetched is not None and not fetched.empty and "candidate_id" in fetched.columns
        else set()
    )
    outcomes: dict[str, str] = {}
    for cid in candidate_ids:
        reported = statuses.get(str(cid))
        if reported in SED_CACHE_RETRYABLE_STATUSES:
            outcomes[str(cid)] = "partial" if str(cid) in row_ids else "error"
        elif str(cid) in row_ids:
            outcomes[str(cid)] = (
                str(reported)
                if reported in SED_CACHE_TERMINAL_STATUSES
                else "hit"
            )
        elif reported in SED_CACHE_TERMINAL_STATUSES:
            outcomes[str(cid)] = str(reported)
        else:
            # An adapter that returns no row and no explicit status did not
            # prove a catalog non-match.  Keep it retryable.
            outcomes[str(cid)] = "error"
    return outcomes


def _fetch_sed_source_with_cache(
    source_key: str,
    fetcher: Callable,
    df: pd.DataFrame,
    progress_callback: ProgressCallback | None = None,
    *,
    chunk_size: int = SED_FETCH_CHUNK_SIZE,
    max_attempts: int = SED_FETCH_MAX_ATTEMPTS,
    retry_base_seconds: float = SED_FETCH_RETRY_BASE_SECONDS,
) -> pd.DataFrame:
    if source_key in SED_CACHE_SKIP_SOURCES:
        return fetcher(df, progress_callback=progress_callback)

    candidate_ids = _sed_candidate_ids(df)
    requested_ids = set(candidate_ids.astype(str))
    astrometry_hashes = _expected_astrometry_hashes(df)
    cache = _read_sed_source_cache(source_key)
    if not cache.empty:
        signature_mask = _cache_signature_mask(cache, source_key, astrometry_hashes)
        cache = cache.loc[signature_mask].copy()
    if not cache.empty:
        cache_status = (
            cache["_cache_status"].fillna("hit").astype(str).str.strip().str.lower()
            if "_cache_status" in cache.columns
            else pd.Series("hit", index=cache.index)
        )
        cached_ids = set(cache.loc[cache_status.isin(SED_CACHE_TERMINAL_STATUSES), "_cache_candidate_id"].astype(str))
    else:
        cached_ids = set()
    hit_cache = pd.DataFrame(columns=CANONICAL_SED_COLUMNS)
    retry_cache = pd.DataFrame(columns=CANONICAL_SED_COLUMNS)
    if not cache.empty:
        status = (
            cache["_cache_status"].fillna("hit").astype(str).str.strip().str.lower()
            if "_cache_status" in cache.columns
            else pd.Series("hit", index=cache.index)
        )
        cached_rows = cache[
            cache["_cache_candidate_id"].isin(requested_ids)
            & status.isin(SED_CACHE_TERMINAL_STATUSES)
            & cache.get("band", pd.Series(None, index=cache.index)).notna()
        ].copy()
        if not cached_rows.empty:
            hit_cache = cached_rows.reindex(columns=CANONICAL_SED_COLUMNS)
        retry_rows = cache[
            cache["_cache_candidate_id"].isin(requested_ids)
            & status.isin(SED_CACHE_RETRYABLE_STATUSES)
            & cache.get("band", pd.Series(None, index=cache.index)).notna()
        ].copy()
        if not retry_rows.empty:
            retry_cache = retry_rows.reindex(columns=CANONICAL_SED_COLUMNS)

    missing_ids = requested_ids - cached_ids
    if progress_callback and cached_ids:
        progress_callback(f"[SED] {source_key} cache hit: {len(requested_ids) - len(missing_ids)}/{len(requested_ids)}")
    if not missing_ids:
        return hit_cache.reset_index(drop=True)

    to_fetch = df.loc[candidate_ids.isin(missing_ids)].copy()
    preclassified_status: dict[str, str] = {}
    if source_key == "ps1" and not to_fetch.empty:
        to_fetch_ids = _sed_candidate_ids(to_fetch)
        dec_values = to_fetch.apply(lambda row: _ra_dec_from_row(row)[1], axis=1)
        outside_mask = dec_values.map(lambda dec: dec is not None and dec < -30.5)
        outside_ids = set(to_fetch_ids.loc[outside_mask].astype(str))
        preclassified_status.update({cid: "outside_footprint" for cid in outside_ids})
        to_fetch = to_fetch.loc[~outside_mask].copy()

    completed_frames: list[pd.DataFrame] = [hit_cache]
    chunk_size = max(int(chunk_size), 1)
    max_attempts = max(int(max_attempts), 1)
    retry_base_seconds = max(float(retry_base_seconds), 0.0)

    # Persist deterministic footprint outcomes even when every remaining row
    # was removed before the network fetch.
    if preclassified_status:
        preclassified_ids = set(preclassified_status)
        _write_sed_source_cache(
            source_key,
            _cache_rows_for_sed_result(
                source_key,
                preclassified_ids,
                pd.DataFrame(columns=CANONICAL_SED_COLUMNS),
                status_by_candidate=preclassified_status,
                astrometry_hashes=astrometry_hashes,
            ),
        )

    n_chunks = max(int(math.ceil(len(to_fetch) / chunk_size)), 1) if len(to_fetch) else 0
    for chunk_number, start in enumerate(range(0, len(to_fetch), chunk_size), start=1):
        chunk = to_fetch.iloc[start:start + chunk_size].copy()
        chunk_ids = set(_sed_candidate_ids(chunk).astype(str))
        accumulated = retry_cache[
            retry_cache["candidate_id"].astype(str).isin(chunk_ids)
        ].copy()
        final_status = {cid: "error" for cid in chunk_ids}
        pending = chunk.copy()

        for attempt in range(1, max_attempts + 1):
            pending_ids = set(_sed_candidate_ids(pending).astype(str))
            try:
                result = fetcher(pending, progress_callback=progress_callback)
                if result is None:
                    explicit = {cid: "error" for cid in pending_ids}
                    fresh = pd.DataFrame(columns=CANONICAL_SED_COLUMNS)
                else:
                    explicit = _fetch_statuses(result)
                    fresh = pd.DataFrame(result).reindex(columns=CANONICAL_SED_COLUMNS)
            except Exception as exc:
                explicit = {cid: "error" for cid in pending_ids}
                fresh = pd.DataFrame(columns=CANONICAL_SED_COLUMNS)
                if progress_callback:
                    progress_callback(
                        f"[SED] {source_key} chunk {chunk_number}/{n_chunks} "
                        f"attempt {attempt}/{max_attempts} failed: {exc}"
                    )

            accumulated = _merge_sed_measurement_frames((accumulated, fresh))
            outcomes = _candidate_fetch_outcomes(pending_ids, fresh, explicit)
            accumulated_ids = set(accumulated["candidate_id"].dropna().astype(str))
            for cid, status_value in outcomes.items():
                if status_value == "miss" and cid in accumulated_ids:
                    status_value = "hit"
                final_status[cid] = status_value

            retry_ids = {
                cid for cid, status_value in outcomes.items()
                if status_value in SED_CACHE_RETRYABLE_STATUSES
            }
            if not retry_ids or attempt >= max_attempts:
                break
            if progress_callback:
                progress_callback(
                    f"[SED] {source_key} chunk {chunk_number}/{n_chunks}: "
                    f"retrying {len(retry_ids)} candidates after attempt {attempt}"
                )
            if retry_base_seconds > 0:
                time.sleep(retry_base_seconds * (2 ** (attempt - 1)))
            pending_ids_series = _sed_candidate_ids(chunk)
            pending = chunk.loc[pending_ids_series.isin(retry_ids)].copy()

        cache_rows = _cache_rows_for_sed_result(
            source_key,
            chunk_ids,
            accumulated,
            status_by_candidate=final_status,
            astrometry_hashes=astrometry_hashes,
        )
        _write_sed_source_cache(source_key, cache_rows)
        completed_frames.append(accumulated)
        if progress_callback:
            status_counts = pd.Series(final_status, dtype=object).value_counts().to_dict()
            progress_callback(
                f"[SED] {source_key} checkpoint {chunk_number}/{n_chunks}: {status_counts}"
            )

    return _merge_sed_measurement_frames(completed_frames)


def rows_from_candidate_frame(df: pd.DataFrame, progress_callback: ProgressCallback | None = None) -> pd.DataFrame:
    rows = []
    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        if progress_callback and (idx == 1 or idx % 1000 == 0 or idx == total):
            progress_callback(f"[SED] payload {idx}/{total}")
        payload = row.to_dict()
        payload_rows = rows_from_payload(
            payload,
            candidate_id=_candidate_id_for_row(row),
            extinction_mode="observed",
        )
        # AllWISE is acquired canonically from IRSA.  Payload copies remain
        # readable as legacy provenance, but are never an acquisition fallback.
        if not payload_rows.empty:
            payload_rows = payload_rows[
                payload_rows["source"].astype(str) != "AllWISE"
            ].copy()
        rows.append(payload_rows)
    if not rows:
        return pd.DataFrame(columns=CANONICAL_SED_COLUMNS)
    if not any(not part.empty for part in rows):
        return pd.DataFrame(columns=CANONICAL_SED_COLUMNS)
    return pd.DataFrame(
        [record for part in rows for record in part.to_dict("records")],
        columns=CANONICAL_SED_COLUMNS,
    )


def _try_requests_get_csv(url: str, timeout: int = 30) -> pd.DataFrame:
    import requests

    res = requests.get(url, timeout=timeout)
    res.raise_for_status()
    if not res.text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(res.text))


def _sleep_after_sed_request() -> None:
    """Apply one configurable, process-local courtesy delay per remote request."""
    if SED_REQUEST_INTERVAL_SECONDS > 0:
        time.sleep(SED_REQUEST_INTERVAL_SECONDS)


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


def _row_float_value(row: pd.Series, aliases: str | Iterable[str] | None) -> float | None:
    if aliases is None:
        return None
    names = [aliases] if isinstance(aliases, str) else list(aliases)
    for name in names:
        value = _safe_float(_row_value(row, name))
        if value is not None:
            return value
    return None


def _row_has_any_column(row: pd.Series, aliases: str | Iterable[str] | None) -> bool:
    if aliases is None:
        return False
    names = [aliases] if isinstance(aliases, str) else list(aliases)
    normalized = {
        str(column).casefold().replace("_", "").replace("-", "")
        for column in row.index
    }
    return any(
        str(name).casefold().replace("_", "").replace("-", "") in normalized
        for name in names
    )


def _catalog_mag_value(row: pd.Series, aliases: str | Iterable[str] | None) -> float | None:
    """Return a plausible catalog magnitude while rejecting common sentinels."""
    value = _row_float_value(row, aliases)
    if value is None or value <= -30.0 or value >= 60.0:
        return None
    return value


def _catalog_mag_error(row: pd.Series, aliases: str | Iterable[str] | None) -> float | None:
    value = _row_float_value(row, aliases)
    if value is None or value <= 0.0 or value >= 10.0:
        return None
    return value


def _catalog_quality_flags(row: pd.Series, columns: Iterable[str], static_flags: str = "") -> str:
    flags = [flag for flag in str(static_flags or "").split(";") if flag]
    for column in columns:
        value = _row_value(row, column)
        if value is None or np.ma.is_masked(value):
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        text = str(value).strip().replace(";", ",")
        if text and text.lower() not in {"nan", "none", "--"}:
            flags.append(f"{column}={text}")
    return ";".join(flags)


def _ra_dec_from_row(row: pd.Series) -> tuple[float | None, float | None]:
    ra = _safe_float(_row_value(row, ("ra", "ra_deg", "RA", "RAJ2000", "RA_ICRS", "RAICRS")))
    dec = _safe_float(_row_value(row, ("dec", "dec_deg", "DEC", "DEJ2000", "DE_ICRS", "DEICRS")))
    return ra, dec


def _fetch_result(rows: list[dict], statuses: Mapping[str, str]) -> pd.DataFrame:
    return _set_fetch_statuses(pd.DataFrame(rows, columns=CANONICAL_SED_COLUMNS), statuses)


def _all_candidate_status(df: pd.DataFrame, status: str) -> dict[str, str]:
    return {str(cid): str(status) for cid in _sed_candidate_ids(df)}


def query_ps1_mean_photometry(
    df: pd.DataFrame,
    radius_arcsec: float = 1.5,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Fetch PS1 mean photometry, using bulk VizieR XMatch for large batches."""
    if len(df) >= SED_BULK_XMATCH_MIN_CANDIDATES:
        spec = VIZIER_SOURCE_SPECS["ps1"]
        matches, statuses = _bulk_vizier_matches(
            df,
            catalog=spec.catalog,
            radius_arcsec=radius_arcsec,
            source_key="ps1",
            progress_callback=progress_callback,
        )
        rows: list[dict] = []
        payloads = _candidate_rows_by_id(df)
        for cid, candidate_matches in matches.items():
            payload_row = payloads.get(cid)
            if payload_row is None or not candidate_matches:
                continue
            separation, selected = candidate_matches[0]
            candidate_rows = _rows_from_vizier_match(
                cid,
                payload_row.to_dict(),
                selected,
                source_key="ps1",
                spec=spec,
                separation_arcsec=separation,
                candidate_separations_arcsec=[item[0] for item in candidate_matches],
            )
            if not candidate_rows and not _vizier_measurement_schema_present(selected, spec):
                statuses[cid] = "error"
            rows.extend(candidate_rows)
        return _fetch_result(rows, statuses)

    rows: list[dict] = []
    statuses: dict[str, str] = {}
    total = len(df)
    for idx, (_, item) in enumerate(df.iterrows(), start=1):
        if progress_callback and (idx == 1 or idx % 500 == 0 or idx == total):
            progress_callback(f"[SED] ps1 {idx}/{total}")
        cid = _candidate_id_for_row(item)
        ra, dec = _ra_dec_from_row(item)
        if ra is None or dec is None:
            statuses[cid] = "error"
            continue
        if dec < -30.5:
            statuses[cid] = "outside_footprint"
            continue
        payload = item.to_dict()
        radius_deg = float(radius_arcsec) / 3600.0
        try:
            url = (
                "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean.csv"
                f"?ra={ra}&dec={dec}&radius={radius_deg}&pagesize=20&format=csv"
            )
            result = _try_requests_get_csv(url)
            if result.empty:
                statuses[cid] = "miss"
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
            if sep is not None and sep <= radius_deg * 1.1:
                sep *= 3600.0
            rows.extend(_rows_from_simple_mag_dict(cid, values, source="Pan-STARRS", distance_pc=distance_pc_from_payload(payload), sep_arcsec=sep))
        except Exception:
            statuses[cid] = "error"
            continue
        finally:
            _sleep_after_sed_request()
    return _fetch_result(rows, statuses)


def query_gaia_gspc_photometry(
    df: pd.DataFrame,
    chunk_size: int = 1000,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Fetch Gaia DR3 GSPC synthetic SDSS and PS1-y photometry, best effort."""
    ids: list[str] = []
    id_to_records: dict[str, list[tuple[str, dict]]] = {}
    statuses = _all_candidate_status(df, "miss")
    for _, item in df.iterrows():
        cid = _candidate_id_for_row(item)
        sid = None
        for id_col in ("gaia_id", "source_id"):
            candidate_sid = _normalize_integer_id(item.get(id_col))
            if candidate_sid:
                sid = candidate_sid
                break
        if sid:
            if sid not in id_to_records:
                ids.append(sid)
            id_to_records.setdefault(sid, []).append((cid, item.to_dict()))
    if not ids:
        return _fetch_result([], statuses)
    rows: list[dict] = []
    try:
        import pyvo
        tap = pyvo.dal.TAPService("https://gea.esac.esa.int/tap-server/tap")
    except Exception as exc:
        if progress_callback:
            progress_callback(f"[SED] gaia_gspc failed: {exc}")
        return _fetch_result([], _all_candidate_status(df, "error"))

    def mag_err_from_flux(row: object, prefix: str) -> float | None:
        flux = _safe_float(row[f"{prefix}_flux"])
        flux_err = _safe_float(row[f"{prefix}_flux_error"])
        if flux is None or flux <= 0 or flux_err is None or flux_err <= 0:
            return None
        return flux_err / flux / (0.4 * math.log(10.0))

    for start in range(0, len(ids), max(int(chunk_size), 1)):
        if progress_callback:
            progress_callback(f"[SED] gaia_gspc {min(start + chunk_size, len(ids))}/{len(ids)}")
        chunk = ids[start:start + max(int(chunk_size), 1)]
        id_list = ",".join(chunk)
        query = (
            "SELECT source_id, "
            "u_sdss_mag, u_sdss_flux, u_sdss_flux_error, "
            "g_sdss_mag, g_sdss_flux, g_sdss_flux_error, "
            "r_sdss_mag, r_sdss_flux, r_sdss_flux_error, "
            "i_sdss_mag, i_sdss_flux, i_sdss_flux_error, "
            "z_sdss_mag, z_sdss_flux, z_sdss_flux_error, "
            "y_ps1_mag, y_ps1_flux, y_ps1_flux_error "
            "FROM gaiadr3.synthetic_photometry_gspc "
            f"WHERE source_id IN ({id_list})"
        )
        try:
            table = tap.search(query).to_table()
        except Exception as exc:
            for sid in chunk:
                for cid, _payload in id_to_records.get(sid, []):
                    statuses[cid] = "error"
            if progress_callback:
                progress_callback(f"[SED] gaia_gspc batch failed: {exc}")
            continue
        finally:
            _sleep_after_sed_request()

        for row in table:
            sid = _normalize_integer_id(row["source_id"])
            if not sid:
                continue
            values = {
                "SDSS_u": (_safe_float(row["u_sdss_mag"]), mag_err_from_flux(row, "u_sdss")),
                "SDSS_g": (_safe_float(row["g_sdss_mag"]), mag_err_from_flux(row, "g_sdss")),
                "SDSS_r": (_safe_float(row["r_sdss_mag"]), mag_err_from_flux(row, "r_sdss")),
                "SDSS_i": (_safe_float(row["i_sdss_mag"]), mag_err_from_flux(row, "i_sdss")),
                "SDSS_z": (_safe_float(row["z_sdss_mag"]), mag_err_from_flux(row, "z_sdss")),
                "PS1_y": (_safe_float(row["y_ps1_mag"]), mag_err_from_flux(row, "y_ps1")),
            }
            for cid, payload in id_to_records.get(sid, []):
                rows.extend(
                    _rows_from_simple_mag_dict(
                        cid,
                        values,
                        source="Gaia GSPC",
                        distance_pc=distance_pc_from_payload(payload),
                        quality_flags="synthetic_from_gaia_xp",
                    )
                )
    return _fetch_result(rows, statuses)


def _gaia_xp_table_source_id(table: object, product_key: object, known_ids: set[str]) -> str | None:
    colnames = list(getattr(table, "colnames", []))
    for column in ("source_id", "SOURCE_ID", "sourceid"):
        if column in colnames and len(table):
            sid = _normalize_integer_id(table[column][0])
            if sid:
                return sid
    meta = getattr(table, "meta", {}) or {}
    for key in ("source_id", "SOURCE_ID", "sourceid"):
        sid = _normalize_integer_id(meta.get(key))
        if sid:
            return sid
    for token in re.findall(r"\d{10,20}", str(product_key)):
        sid = _normalize_integer_id(token)
        if sid in known_ids:
            return sid
    return next(iter(known_ids)) if len(known_ids) == 1 else None


def _gaia_xp_column_values(table: object, column: str, target_unit: u.UnitBase, default_unit: u.UnitBase) -> np.ndarray:
    values = np.asarray(table[column], dtype=float)
    source_unit = getattr(table[column], "unit", None)
    try:
        unit = u.Unit(source_unit) if source_unit else default_unit
        return (values * unit).to_value(target_unit)
    except Exception:
        return (values * default_unit).to_value(target_unit)


def query_gaia_xp_sampled(
    df: pd.DataFrame,
    chunk_size: int = GAIA_XP_FETCH_CHUNK_SIZE,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Fetch Gaia DR3 XP sampled spectra and retain samples as correlated SED rows."""
    ids: list[str] = []
    id_to_payload: dict[str, dict] = {}
    id_to_candidate: dict[str, str] = {}
    statuses = _all_candidate_status(df, "miss")
    for _, item in df.iterrows():
        cid = _candidate_id_for_row(item)
        sid = None
        for id_col in ("gaia_id", "source_id"):
            candidate_sid = _normalize_integer_id(item.get(id_col))
            if candidate_sid:
                sid = candidate_sid
                break
        if sid:
            ids.append(sid)
            id_to_payload[sid] = item.to_dict()
            id_to_candidate[sid] = cid
    if not ids:
        return _fetch_result([], statuses)

    try:
        from astroquery.gaia import Gaia
    except Exception:
        return _fetch_result([], _all_candidate_status(df, "error"))

    rows: list[dict] = []
    flux_lambda_unit = u.erg / u.s / u.cm**2 / u.AA
    for start in range(0, len(ids), max(int(chunk_size), 1)):
        chunk = ids[start:start + max(int(chunk_size), 1)]
        if progress_callback:
            progress_callback(f"[SED] gaia_xp {min(start + len(chunk), len(ids))}/{len(ids)}")
        try:
            products_by_key = Gaia.load_data(
                ids=[int(source_id) for source_id in chunk],
                retrieval_type="XP_SAMPLED",
                data_release="Gaia DR3",
            )
        except Exception as exc:
            if progress_callback and start == 0:
                progress_callback(f"[SED] gaia_xp first batch failed: {exc}")
            for sid in chunk:
                statuses[id_to_candidate.get(sid, sid)] = "error"
            continue
        finally:
            _sleep_after_sed_request()
        if not products_by_key:
            continue

        known_ids = set(chunk)
        parse_error = False
        produced_ids: set[str] = set()
        for product_key, products in products_by_key.items():
            product_list = products if isinstance(products, (list, tuple)) else [products]
            for product in product_list:
                try:
                    table = product.to_table() if hasattr(product, "to_table") else product
                    colnames = list(getattr(table, "colnames", []))
                    if not {"wavelength", "flux"}.issubset(colnames) or len(table) == 0:
                        parse_error = True
                        continue
                    sid = _gaia_xp_table_source_id(table, product_key, known_ids)
                    if sid is None:
                        parse_error = True
                        continue
                    payload = id_to_payload.get(sid, {})
                    cid = str(payload.get("candidate_id") or payload.get("asas_sn_id") or sid)
                    wavelength = _gaia_xp_column_values(table, "wavelength", u.AA, u.nm)
                    flux_lambda = _gaia_xp_column_values(
                        table,
                        "flux",
                        flux_lambda_unit,
                        u.W / u.m**2 / u.nm,
                    )
                    if "flux_error" in colnames:
                        flux_lambda_err = _gaia_xp_column_values(
                            table,
                            "flux_error",
                            flux_lambda_unit,
                            u.W / u.m**2 / u.nm,
                        )
                    else:
                        flux_lambda_err = np.full_like(flux_lambda, np.nan)
                    distance_pc = distance_pc_from_payload(payload)
                    for sample_index, (lam, flam, flam_err) in enumerate(zip(wavelength, flux_lambda, flux_lambda_err, strict=False)):
                        if not np.isfinite(lam) or not np.isfinite(flam) or lam <= 0 or flam <= 0:
                            continue
                        try:
                            fnu = (flam * flux_lambda_unit).to_value(
                                u.Jy,
                                equivalencies=u.spectral_density(lam * u.AA),
                            )
                            fnu_err = (
                                (flam_err * flux_lambda_unit).to_value(
                                    u.Jy,
                                    equivalencies=u.spectral_density(lam * u.AA),
                                )
                                if np.isfinite(flam_err) and flam_err > 0
                                else None
                            )
                        except Exception:
                            continue
                        if not np.isfinite(fnu) or fnu <= 0:
                            continue
                        mag_err = (
                            fnu_err / fnu / (0.4 * math.log(10.0))
                            if fnu_err is not None and fnu_err > 0
                            else None
                        )
                        bp = SedBandpass(
                            source="Gaia XP",
                            band=f"XP_{sample_index:03d}_{lam:.1f}A",
                            mag_col=None,
                            err_col=None,
                            mag_system="Jy",
                            lambda_eff_angstrom=float(lam),
                            fnu_zero_jy=None,
                        )
                        row = _row_from_bandpass(
                            candidate_id=cid,
                            bandpass=bp,
                            mag=float(fnu),
                            mag_err=mag_err,
                            distance_pc=distance_pc,
                            av=None,
                            dereddened=False,
                            quality_flags="correlated_spectrum;gaia_xp_sampled",
                        )
                        if row is None:
                            continue
                        row["flux_lambda"] = float(flam)
                        row["flux_lambda_err"] = float(flam_err) if np.isfinite(flam_err) and flam_err > 0 else None
                        row["flux_nu_jy"] = float(fnu)
                        row["flux_nu_jy_err"] = float(fnu_err) if fnu_err is not None else None
                        row["lambda_l_lambda"] = lambda_l_lambda_from_flux_lambda(flam, lam, distance_pc)
                        if row["lambda_l_lambda"] is not None and row["flux_lambda_err"] is not None:
                            row["lambda_l_lambda_err"] = row["lambda_l_lambda"] * row["flux_lambda_err"] / flam
                        rows.append(row)
                        produced_ids.add(cid)
                except Exception as exc:
                    parse_error = True
                    if progress_callback and start == 0:
                        progress_callback(f"[SED] gaia_xp product failed: {exc}")
                    continue
        if parse_error:
            for sid in chunk:
                cid = id_to_candidate.get(sid, sid)
                if cid not in produced_ids:
                    statuses[cid] = "error"
    return _fetch_result(rows, statuses)


@dataclass(frozen=True)
class VizierSourceSpec:
    source: str
    catalog: str
    radius_arcsec: float
    ra_col: str
    dec_col: str
    bands: dict[str, tuple[str | tuple[str, ...], str | tuple[str, ...] | None]]
    quality_cols: tuple[str, ...] = ()
    static_flags: str = ""


VIZIER_SOURCE_SPECS: dict[str, VizierSourceSpec] = {
    "ps1": VizierSourceSpec("Pan-STARRS", "II/349/ps1", 1.5, "RAJ2000", "DEJ2000", {
        "g": ("gmag", "e_gmag"),
        "r": ("rmag", "e_rmag"),
        "i": ("imag", "e_imag"),
        "z": ("zmag", "e_zmag"),
        "y": ("ymag", "e_ymag"),
    }, quality_cols=("Qual", "Nd", "Ns", "gFlags", "rFlags", "iFlags", "zFlags", "yFlags")),
    "galex": VizierSourceSpec("GALEX", "II/335/galex_ais", 3.0, "RAJ2000", "DEJ2000", {
        "FUV": (("FUVmag", "FUV"), ("e_FUVmag", "e_FUV")),
        "NUV": (("NUVmag", "NUV"), ("e_NUVmag", "e_NUV")),
    }, quality_cols=("Fafl", "Nafl", "Fexf", "Nexf", "G", "N")),
    "catwise": VizierSourceSpec("CatWISE2020", "II/365/catwise", 2.75, "RAPMdeg", "DEPMdeg", {
        "W1": ("W1mproPM", "e_W1mproPM"),
        "W2": ("W2mproPM", "e_W2mproPM"),
    }, quality_cols=("pmQual", "ccf", "abf", "k1", "k2", "snrW1pm", "snrW2pm", "chi2W1pm", "chi2W2pm")),
    "sdss": VizierSourceSpec("SDSS", "V/154/sdss16", 1.5, "RA_ICRS", "DE_ICRS", {
        "u": ("umag", "e_umag"), "g": ("gmag", "e_gmag"), "r": ("rmag", "e_rmag"), "i": ("imag", "e_imag"), "z": ("zmag", "e_zmag"),
    }),
    "skymapper": VizierSourceSpec("SkyMapper", "II/379/smssdr4", 1.5, "RAICRS", "DEICRS", {
        "u": ("uPSF", "e_uPSF"), "v": ("vPSF", "e_vPSF"), "g": ("gPSF", "e_gPSF"), "r": ("rPSF", "e_rPSF"), "i": ("iPSF", "e_iPSF"), "z": ("zPSF", "e_zPSF"),
    }),
    "des": VizierSourceSpec("DES", "II/371/des_dr2", 1.2, "RA_ICRS", "DE_ICRS", {
        "g": (("gmagPSF", "WAVG_MAG_PSF_G", "MAG_PSF_G", "gmag"), ("e_gmagPSF", "WAVG_MAGERR_PSF_G", "MAGERR_PSF_G", "e_gmag")),
        "r": (("rmagPSF", "WAVG_MAG_PSF_R", "MAG_PSF_R", "rmag"), ("e_rmagPSF", "WAVG_MAGERR_PSF_R", "MAGERR_PSF_R", "e_rmag")),
        "i": (("imagPSF", "WAVG_MAG_PSF_I", "MAG_PSF_I", "imag"), ("e_imagPSF", "WAVG_MAGERR_PSF_I", "MAGERR_PSF_I", "e_imag")),
        "z": (("zmagPSF", "WAVG_MAG_PSF_Z", "MAG_PSF_Z", "zmag"), ("e_zmagPSF", "WAVG_MAGERR_PSF_Z", "MAGERR_PSF_Z", "e_zmag")),
        "Y": (("YmagPSF", "WAVG_MAG_PSF_Y", "MAG_PSF_Y", "Ymag"), ("e_YmagPSF", "WAVG_MAGERR_PSF_Y", "MAGERR_PSF_Y", "e_Ymag")),
    }),
    "ukidss": VizierSourceSpec("UKIDSS", "II/319/las9", 1.2, "RAJ2000", "DEJ2000", {
        "Y": ("Ymag", "e_Ymag"), "J": (("Jmag1", "Jmag2"), ("e_Jmag1", "e_Jmag2")), "H": ("Hmag", "e_Hmag"), "K": ("Kmag", "e_Kmag"),
    }, quality_cols=(
        "mergedClass", "pStar", "YppErrBits", "J1ppErrBits", "J2ppErrBits",
        "HppErrBits", "KppErrBits",
    )),
    "vista": VizierSourceSpec("VISTA/VVV", "II/376/vvv4", 1.2, "RAJ2000", "DEJ2000", {
        "Z": (("Z1ap3", "Z2ap3", "Zmag3"), ("e_Z1ap3", "e_Z2ap3", "e_Zmag3")),
        "Y": (("Y1ap3", "Y2ap3", "Ymag3"), ("e_Y1ap3", "e_Y2ap3", "e_Ymag3")),
        "J": (("J1ap3", "J2ap3", "Jmag3"), ("e_J1ap3", "e_J2ap3", "e_Jmag3")),
        "H": (("H1ap3", "H2ap3", "Hmag3"), ("e_H1ap3", "e_H2ap3", "e_Hmag3")),
        "Ks": (("Ks1ap3", "Ks2ap3", "Ksmag3"), ("e_Ks1ap3", "e_Ks2ap3", "e_Ksmag3")),
    }, quality_cols=(
        "Mclass", "pStar", "ZppErrBits", "YppErrBits", "JppErrBits",
        "HppErrBits", "KsppErrBits",
    )),
    "vhs": VizierSourceSpec("VISTA/VHS", "II/367/vhs_dr5", 1.2, "RAJ2000", "DEJ2000", {
        "Y": ("Yap3", "e_Yap3"),
        "J": ("Jap3", "e_Jap3"),
        "H": ("Hap3", "e_Hap3"),
        "Ks": ("Ksap3", "e_Ksap3"),
    }, quality_cols=("PriOrSec", "Mclass", "pStar", "Yperrbits", "Jperrbits", "Hperrbits", "Ksperrbits")),
    "viking": VizierSourceSpec("VISTA/VIKING", "II/382/viking4", 1.2, "RAJ2000", "DEJ2000", {
        "Z": ("Zap3", "e_Zap3"),
        "Y": ("Yap3", "e_Yap3"),
        "J": (("Jap3", "J1ap3", "J2ap3"), ("e_Jap3", "e_J1ap3", "e_J2ap3")),
        "H": ("Hap3", "e_Hap3"),
        "Ks": ("Ksap3", "e_Ksap3"),
    }, quality_cols=(
        "PriOrSec", "Mclass", "pStar", "Zperrbits", "Yperrbits",
        "Jperrbits", "Hperrbits", "Ksperrbits",
    )),
    "vphas": VizierSourceSpec("VPHAS+", "II/386/vphasplus32", 1.0, "RAJ2000", "DEJ2000", {
        "u": ("uap3", "e_uap3"),
        "g": ("gap3", "e_gap3"),
        "r": ("rap3", "e_rap3"),
        "i": ("iap3", "e_iap3"),
        "Halpha": ("Haap3", "e_Haap3"),
    }),
    "swift_uvot": VizierSourceSpec("Swift/UVOT", "II/339/uvotssc1", 2.5, "RAJ2000", "DEJ2000", {
        "UVW2": ("UVW2-AB", "e_UVW2"),
        "UVM2": ("UVM2-AB", "e_UVM2"),
        "UVW1": ("UVW1-AB", "e_UVW1"),
        "U": ("U-AB", "e_Umag"),
        "B": ("B-AB", "e_Bmag"),
        "V": ("V-AB", "e_Vmag"),
    }, quality_cols=("Nd", "fUVW2", "fUVM2", "fUVW1", "fU", "fB", "fV"), static_flags="non_simultaneous_pointed"),
    "xmm_om": VizierSourceSpec("XMM-OM", "II/378/xmmom6s", 2.5, "RAJ2000", "DEJ2000", {
        "UVW2": ("UVW2mAB", "e_UVW2mAB"),
        "UVM2": ("UVM2mAB", "e_UVM2mAB"),
        "UVW1": ("UVW1mAB", "e_UVW1mAB"),
        "U": ("UmAB", "e_UmAB"),
        "B": ("BmAB", "e_BmAB"),
        "V": ("VmAB", "e_VmAB"),
    }, quality_cols=("Nobs", "q.UVW2", "q.UVM2", "q.UVW1", "q.U", "q.B", "q.V"), static_flags="non_simultaneous_pointed"),
}


def _candidate_rows_by_id(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {_candidate_id_for_row(row): row for _, row in df.iterrows()}


def _candidate_total_proper_motion_masyr(payload: Mapping[str, object]) -> float | None:
    pmra = _safe_float(_row_value(pd.Series(payload), ("pmra", "gaia_pmra")))
    pmdec = _safe_float(_row_value(pd.Series(payload), ("pmdec", "gaia_pmdec")))
    if pmra is None or pmdec is None:
        return None
    return float(math.hypot(pmra, pmdec))


def _catalog_gaia_id(row: pd.Series) -> str | None:
    value = _row_value(
        row,
        ("GaiaDR3", "GaiaEDR3", "GaiaDR2", "gaia_dr3_source_id", "gaia_source_id"),
    )
    return _normalize_integer_id(value)


def _counterpart_validation_flags(
    payload: Mapping[str, object],
    catalog_row: pd.Series,
    *,
    source_key: str,
    separation_arcsec: float | None,
    candidate_separations_arcsec: Iterable[float],
    radius_arcsec: float,
) -> list[str]:
    separations = sorted(
        float(value)
        for value in candidate_separations_arcsec
        if value is not None and np.isfinite(float(value))
    )
    flags: list[str] = []
    if len(separations) > 1:
        flags.extend(("multiple_catalog_matches", f"match_count={len(separations)}"))
        second = separations[1]
        flags.append(f"second_nearest_sep_arcsec={second:.4f}")
        if second - separations[0] <= SED_MATCH_AMBIGUITY_GAP_ARCSEC:
            flags.extend(("ambiguous_counterpart", "bad_quality"))
    if (
        separation_arcsec is None
        or not np.isfinite(float(separation_arcsec))
    ):
        flags.extend(("match_separation_unavailable", "bad_quality"))
    elif float(separation_arcsec) > SED_MATCH_EDGE_FRACTION * float(radius_arcsec):
        flags.extend(("large_match_separation", "bad_quality"))

    candidate_gaia_id = None
    for column in ("gaia_id", "source_id"):
        candidate_gaia_id = _normalize_integer_id(payload.get(column))
        if candidate_gaia_id:
            break
    catalog_gaia_id = _catalog_gaia_id(catalog_row)
    if candidate_gaia_id and catalog_gaia_id and candidate_gaia_id != catalog_gaia_id:
        flags.extend(("gaia_id_conflict", "bad_quality"))

    total_pm = _candidate_total_proper_motion_masyr(payload)
    if (
        total_pm is not None
        and total_pm >= SED_MATCH_HIGH_PM_MASYR
        and not (candidate_gaia_id and catalog_gaia_id == candidate_gaia_id)
        and source_key != "catwise"
    ):
        flags.extend(
            (
                "proper_motion_sensitive_match",
                f"pm_total_masyr={total_pm:.3f}",
                "bad_quality",
            )
        )
    return list(dict.fromkeys(flags))


def _bulk_vizier_matches(
    df: pd.DataFrame,
    *,
    catalog: str,
    radius_arcsec: float,
    source_key: str,
    progress_callback: ProgressCallback | None = None,
) -> tuple[dict[str, list[tuple[float, pd.Series]]], dict[str, str]]:
    """Crossmatch one candidate chunk against a VizieR table in one CDS job."""
    statuses = _all_candidate_status(df, "miss")
    target_rows: list[dict[str, object]] = []
    for _, item in df.iterrows():
        cid = _candidate_id_for_row(item)
        ra, dec = _ra_dec_from_row(item)
        if ra is None or dec is None:
            statuses[cid] = "error"
            continue
        target_rows.append(
            {
                "malca_candidate_id": cid,
                "malca_ra": float(ra),
                "malca_dec": float(dec),
            }
        )
    if not target_rows:
        return {}, statuses

    try:
        from astropy.table import Table
        from astroquery.xmatch import XMatch

        targets = Table.from_pandas(pd.DataFrame(target_rows))
        XMatch.TIMEOUT = max(float(getattr(XMatch, "TIMEOUT", 60.0)), 120.0)
        if progress_callback:
            progress_callback(
                f"[SED] {source_key} CDS XMatch: {len(target_rows)} targets against {catalog}"
            )
        result = XMatch.query(
            cat1=targets,
            cat2=f"vizier:{catalog}",
            max_distance=float(radius_arcsec) * u.arcsec,
            colRA1="malca_ra",
            colDec1="malca_dec",
            cache=False,
        )
        frame = result.to_pandas()
    except Exception as exc:
        for target in target_rows:
            statuses[str(target["malca_candidate_id"])] = "error"
        if progress_callback:
            progress_callback(f"[SED] {source_key} CDS XMatch failed: {exc}")
        return {}, statuses
    finally:
        _sleep_after_sed_request()

    if frame.empty:
        return {}, statuses
    candidate_col = next(
        (column for column in frame.columns if str(column).casefold() == "malca_candidate_id"),
        None,
    )
    separation_col = next(
        (
            column
            for column in frame.columns
            if str(column).strip().casefold() in {"angdist", "_r"}
        ),
        None,
    )
    if candidate_col is None or separation_col is None:
        for target in target_rows:
            statuses[str(target["malca_candidate_id"])] = "error"
        return {}, statuses

    matches: dict[str, list[tuple[float, pd.Series]]] = {}
    for _, row in frame.iterrows():
        cid = _clean_text(row.get(candidate_col))
        separation = _safe_float(row.get(separation_col))
        if not cid or separation is None or separation > float(radius_arcsec):
            continue
        matches.setdefault(cid, []).append((float(separation), row))
    for candidate_matches in matches.values():
        candidate_matches.sort(key=lambda item: item[0])
    return matches, statuses


def _rows_from_vizier_match(
    candidate_id: str,
    payload: Mapping[str, object],
    row: pd.Series,
    *,
    source_key: str,
    spec: VizierSourceSpec,
    separation_arcsec: float | None,
    candidate_separations_arcsec: Iterable[float],
) -> list[dict]:
    values = {
        band: (
            _catalog_mag_value(row, mag_col),
            _catalog_mag_error(row, err_col) if err_col else None,
        )
        for band, (mag_col, err_col) in spec.bands.items()
    }
    validation_flags = _counterpart_validation_flags(
        payload,
        row,
        source_key=source_key,
        separation_arcsec=separation_arcsec,
        candidate_separations_arcsec=candidate_separations_arcsec,
        radius_arcsec=spec.radius_arcsec,
    )
    catalog_flags = _catalog_quality_flags(row, spec.quality_cols, spec.static_flags)
    quality_flags = ";".join(
        [flag for flag in (catalog_flags, *validation_flags) if str(flag).strip()]
    )
    output = _rows_from_simple_mag_dict(
        candidate_id,
        values,
        source=spec.source,
        distance_pc=distance_pc_from_payload(payload),
        sep_arcsec=separation_arcsec,
        quality_flags=quality_flags,
    )
    for measurement in output:
        native_flags = _vizier_native_quality_flags(
            source_key,
            row,
            str(measurement.get("band") or ""),
        )
        if native_flags:
            existing = [
                flag for flag in str(measurement.get("quality_flags") or "").split(";")
                if flag
            ]
            measurement["quality_flags"] = ";".join(
                dict.fromkeys([*existing, *native_flags])
            )
            if "bad_quality" in native_flags:
                measurement["fit_policy"] = "diagnostic_only"
    return output


def _vizier_native_quality_flags(
    source_key: str,
    row: pd.Series,
    band: str,
) -> list[str]:
    """Translate documented catalog diagnostics into a common reject flag."""
    key = str(source_key).strip().lower()
    band_text = str(band).strip()
    flags: list[str] = []

    if key == "catwise":
        band_index = {"W1": 0, "W2": 1}.get(band_text)
        for column, flag_name in (("ccf", "catwise_artifact"), ("abf", "catwise_blend")):
            raw = _clean_text(_row_value(row, column))
            if raw and band_index is not None:
                character = raw[band_index] if band_index < len(raw) else raw
                if character not in {"0", "-"}:
                    flags.extend((flag_name, "bad_quality"))
        snr = _row_float_value(row, f"snr{band_text}pm")
        if snr is not None and snr < 5.0:
            flags.extend(("catwise_low_snr", "bad_quality"))

    if key in {"ukidss", "vista", "vhs", "viking"}:
        morphology = _row_float_value(row, ("Mclass", "mergedClass", "mergedclass"))
        if morphology is not None and int(round(morphology)) not in {-1, -2}:
            flags.extend(("nonstellar_morphology", "bad_quality"))
        p_star = _row_float_value(row, ("pStar", "pstar"))
        if p_star is not None and p_star < 0.8:
            flags.extend(("low_stellar_probability", "bad_quality"))
        band_aliases = {
            "K": ("KppErrBits", "Kperrbits", "Kspperrbits", "Ksperrbits"),
            "Ks": ("KsppErrBits", "Ksperrbits", "Kspperrbits"),
            "J": ("JppErrBits", "Jperrbits", "J1ppErrBits", "J2ppErrBits"),
            "H": ("HppErrBits", "Hperrbits"),
            "Y": ("YppErrBits", "Yperrbits"),
            "Z": ("ZppErrBits", "Zperrbits"),
        }
        error_bits = _row_float_value(row, band_aliases.get(band_text, ()))
        if error_bits is not None and int(error_bits) >= 256:
            flags.extend(("catalog_error_bits_ge_256", "bad_quality"))

    return list(dict.fromkeys(flags))


def _vizier_measurement_schema_present(row: pd.Series, spec: VizierSourceSpec) -> bool:
    return any(
        _row_has_any_column(row, mag_aliases)
        for mag_aliases, _error_aliases in spec.bands.values()
    )


def query_vizier_source(
    df: pd.DataFrame,
    key: str,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Fetch one VizieR-backed source spec by nearest cone match, best effort."""
    spec = VIZIER_SOURCE_SPECS[key]
    rows: list[dict] = []
    statuses: dict[str, str] = _all_candidate_status(df, "miss")

    if len(df) >= SED_BULK_XMATCH_MIN_CANDIDATES:
        matches, statuses = _bulk_vizier_matches(
            df,
            catalog=spec.catalog,
            radius_arcsec=spec.radius_arcsec,
            source_key=key,
            progress_callback=progress_callback,
        )
        payloads = _candidate_rows_by_id(df)
        for cid, candidate_matches in matches.items():
            payload_row = payloads.get(cid)
            if payload_row is None or not candidate_matches:
                continue
            separation, selected = candidate_matches[0]
            candidate_rows = _rows_from_vizier_match(
                cid,
                payload_row.to_dict(),
                selected,
                source_key=key,
                spec=spec,
                separation_arcsec=separation,
                candidate_separations_arcsec=[item[0] for item in candidate_matches],
            )
            if not candidate_rows and not _vizier_measurement_schema_present(selected, spec):
                statuses[cid] = "error"
            rows.extend(candidate_rows)
        return _fetch_result(rows, statuses)

    try:
        from astropy.coordinates import SkyCoord
        from astroquery.vizier import Vizier
    except Exception:
        return _fetch_result([], _all_candidate_status(df, "error"))

    viz = Vizier(columns=["**"], row_limit=5)
    viz.TIMEOUT = VIZIER_QUERY_TIMEOUT_SEC
    total = len(df)
    for idx, (_, item) in enumerate(df.iterrows(), start=1):
        if progress_callback and (idx == 1 or idx % 500 == 0 or idx == total):
            progress_callback(f"[SED] {key} {idx}/{total}")
        cid = _candidate_id_for_row(item)
        ra, dec = _ra_dec_from_row(item)
        if ra is None or dec is None:
            statuses[cid] = "error"
            continue
        payload = item.to_dict()
        try:
            target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            tables = viz.query_region(
                target,
                radius=spec.radius_arcsec * u.arcsec,
                catalog=spec.catalog,
            )
            if not tables:
                statuses[cid] = "miss"
                continue
            result = tables[0].to_pandas()
            if result.empty:
                statuses[cid] = "miss"
                continue
            candidate_matches: list[tuple[float, pd.Series]] = []
            for _, candidate_match in result.iterrows():
                match_ra = _safe_float(
                    _row_value(candidate_match, (spec.ra_col, "RA_ICRS", "RAJ2000", "RAICRS"))
                )
                match_dec = _safe_float(
                    _row_value(candidate_match, (spec.dec_col, "DE_ICRS", "DEJ2000", "DEICRS"))
                )
                if match_ra is None or match_dec is None:
                    continue
                separation = float(
                    target.separation(SkyCoord(ra=match_ra * u.deg, dec=match_dec * u.deg)).arcsec
                )
                if separation <= float(spec.radius_arcsec):
                    candidate_matches.append((separation, candidate_match))
            if not candidate_matches:
                statuses[cid] = "miss"
                continue
            candidate_matches.sort(key=lambda item: item[0])
            sep_arcsec, row = candidate_matches[0]
            candidate_rows = _rows_from_vizier_match(
                cid,
                payload,
                row,
                source_key=key,
                spec=spec,
                separation_arcsec=sep_arcsec,
                candidate_separations_arcsec=[item[0] for item in candidate_matches],
            )
            if not candidate_rows and not _vizier_measurement_schema_present(row, spec):
                statuses[cid] = "error"
            rows.extend(candidate_rows)
        except Exception as exc:
            statuses[cid] = "error"
            if progress_callback and idx == 1:
                progress_callback(f"[SED] {key} first lookup failed: {exc}")
            continue
        finally:
            _sleep_after_sed_request()
    return _fetch_result(rows, statuses)


def _tap_bbox_clause(ra: float, dec: float, radius_deg: float) -> str:
    ra_pad = radius_deg / max(math.cos(math.radians(dec)), 0.01)
    ra_min = ra - ra_pad
    ra_max = ra + ra_pad
    if ra_min < 0:
        ra_clause = f"(ra >= {ra_min + 360.0:.12f} OR ra <= {ra_max:.12f})"
    elif ra_max >= 360.0:
        ra_clause = f"(ra >= {ra_min:.12f} OR ra <= {ra_max - 360.0:.12f})"
    else:
        ra_clause = f"ra BETWEEN {ra_min:.12f} AND {ra_max:.12f}"
    return (
        f"({ra_clause} AND dec BETWEEN {dec - radius_deg:.12f} "
        f"AND {dec + radius_deg:.12f})"
    )


def _bulk_tap_bbox_matches(
    tap: object,
    df: pd.DataFrame,
    *,
    table: str,
    select_columns: str,
    radius_arcsec: float,
    source_key: str,
    progress_callback: ProgressCallback | None = None,
    query_chunk_size: int = 100,
    max_rows: int = 100_000,
) -> tuple[dict[str, list[tuple[float, pd.Series]]], dict[str, str]]:
    """Fetch small sky boxes in TAP batches and crossmatch them locally."""
    statuses = _all_candidate_status(df, "miss")
    valid: list[tuple[str, float, float]] = []
    for _, item in df.iterrows():
        cid = _candidate_id_for_row(item)
        ra, dec = _ra_dec_from_row(item)
        if ra is None or dec is None:
            statuses[cid] = "error"
        else:
            valid.append((cid, float(ra), float(dec)))
    matches: dict[str, list[tuple[float, pd.Series]]] = {}
    radius_deg = float(radius_arcsec) / 3600.0
    query_chunk_size = max(int(query_chunk_size), 1)
    n_chunks = max(int(math.ceil(len(valid) / query_chunk_size)), 1) if valid else 0
    for chunk_number, start in enumerate(range(0, len(valid), query_chunk_size), start=1):
        targets = valid[start:start + query_chunk_size]
        where_clause = " OR ".join(
            _tap_bbox_clause(ra, dec, radius_deg) for _cid, ra, dec in targets
        )
        query = (
            f"SELECT TOP {int(max_rows)} {select_columns} FROM {table} "
            f"WHERE {where_clause}"
        )
        if progress_callback:
            progress_callback(
                f"[SED] {source_key} TAP batch {chunk_number}/{n_chunks}: {len(targets)} targets"
            )
        try:
            result = tap.search(query).to_table().to_pandas()
        except Exception as exc:
            for cid, _ra, _dec in targets:
                statuses[cid] = "error"
            if progress_callback:
                progress_callback(f"[SED] {source_key} TAP batch failed: {exc}")
            continue
        finally:
            _sleep_after_sed_request()
        if len(result) >= int(max_rows):
            for cid, _ra, _dec in targets:
                statuses[cid] = "error"
            if progress_callback:
                progress_callback(
                    f"[SED] {source_key} TAP batch reached TOP {max_rows}; marked retryable"
                )
            continue
        if result.empty:
            continue
        if not {"ra", "dec"}.issubset(result.columns):
            for cid, _ra, _dec in targets:
                statuses[cid] = "error"
            continue
        catalog_ra = pd.to_numeric(result.get("ra"), errors="coerce").to_numpy(dtype=float)
        catalog_dec = pd.to_numeric(result.get("dec"), errors="coerce").to_numpy(dtype=float)
        finite_catalog = np.isfinite(catalog_ra) & np.isfinite(catalog_dec)
        for cid, ra, dec in targets:
            wrapped_dra = ((catalog_ra - ra + 180.0) % 360.0) - 180.0
            mean_dec = 0.5 * (catalog_dec + dec)
            separations = np.hypot(
                wrapped_dra * np.cos(np.deg2rad(mean_dec)),
                catalog_dec - dec,
            ) * 3600.0
            indexes = np.flatnonzero(
                finite_catalog & np.isfinite(separations) & (separations <= float(radius_arcsec))
            )
            if indexes.size:
                candidate_matches = [
                    (float(separations[index]), result.iloc[int(index)])
                    for index in indexes
                ]
                candidate_matches.sort(key=lambda item: item[0])
                matches[cid] = candidate_matches
    return matches, statuses


def _rows_from_decaps_match(
    candidate_id: str,
    payload: Mapping[str, object],
    row: pd.Series,
    *,
    separation_arcsec: float,
    candidate_separations_arcsec: Iterable[float],
    radius_arcsec: float,
) -> list[dict]:
    values = {
        "g": (_row_float_value(row, ("mean_mag_g", "median_mag_g", "mean_cmag_g", "median_cmag_g")), None),
        "r": (_row_float_value(row, ("mean_mag_r", "median_mag_r", "mean_cmag_r", "median_cmag_r")), None),
        "i": (_row_float_value(row, ("mean_mag_i", "median_mag_i", "mean_cmag_i", "median_cmag_i")), None),
        "z": (_row_float_value(row, ("mean_mag_z", "median_mag_z", "mean_cmag_z", "median_cmag_z")), None),
        "Y": (_row_float_value(row, ("mean_mag_y", "median_mag_y", "mean_cmag_y", "median_cmag_y")), None),
    }
    flags = _counterpart_validation_flags(
        payload,
        row,
        source_key="decaps",
        separation_arcsec=separation_arcsec,
        candidate_separations_arcsec=candidate_separations_arcsec,
        radius_arcsec=radius_arcsec,
    )
    return _rows_from_simple_mag_dict(
        candidate_id,
        values,
        source="DECaPS",
        distance_pc=distance_pc_from_payload(payload),
        sep_arcsec=separation_arcsec,
        quality_flags=";".join(flags),
    )


def query_decaps_photometry(
    df: pd.DataFrame,
    radius_arcsec: float = 1.2,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Fetch DECaPS DR2 mean photometry from NOIRLab Astro Data Lab TAP."""
    rows: list[dict] = []
    statuses = _all_candidate_status(df, "miss")
    try:
        import pyvo
        tap = pyvo.dal.TAPService("https://datalab.noirlab.edu/tap")
    except Exception:
        return _fetch_result([], _all_candidate_status(df, "error"))

    if len(df) >= SED_BULK_XMATCH_MIN_CANDIDATES:
        select_columns = (
            "ra, dec, "
            "mean_mag_g, median_mag_g, mean_cmag_g, median_cmag_g, "
            "mean_mag_r, median_mag_r, mean_cmag_r, median_cmag_r, "
            "mean_mag_i, median_mag_i, mean_cmag_i, median_cmag_i, "
            "mean_mag_z, median_mag_z, mean_cmag_z, median_cmag_z, "
            "mean_mag_y, median_mag_y, mean_cmag_y, median_cmag_y"
        )
        matches, statuses = _bulk_tap_bbox_matches(
            tap,
            df,
            table="decaps_dr2.object",
            select_columns=select_columns,
            radius_arcsec=radius_arcsec,
            source_key="decaps",
            progress_callback=progress_callback,
        )
        payloads = _candidate_rows_by_id(df)
        for cid, candidate_matches in matches.items():
            payload_row = payloads.get(cid)
            if payload_row is None or not candidate_matches:
                continue
            separation, selected = candidate_matches[0]
            candidate_rows = _rows_from_decaps_match(
                cid,
                payload_row.to_dict(),
                selected,
                separation_arcsec=separation,
                candidate_separations_arcsec=[item[0] for item in candidate_matches],
                radius_arcsec=radius_arcsec,
            )
            if candidate_rows:
                rows.extend(candidate_rows)
        return _fetch_result(rows, statuses)

    total = len(df)
    radius_deg = float(radius_arcsec) / 3600.0
    for idx, (_, item) in enumerate(df.iterrows(), start=1):
        if progress_callback and (idx == 1 or idx % 500 == 0 or idx == total):
            progress_callback(f"[SED] decaps {idx}/{total}")
        cid = _candidate_id_for_row(item)
        ra, dec = _ra_dec_from_row(item)
        if ra is None or dec is None:
            statuses[cid] = "error"
            continue
        payload = item.to_dict()
        ra_pad = radius_deg / max(math.cos(math.radians(dec)), 0.01)
        ra_min = ra - ra_pad
        ra_max = ra + ra_pad
        if ra_min < 0:
            ra_clause = f"(ra >= {ra_min + 360.0} OR ra <= {ra_max})"
        elif ra_max >= 360.0:
            ra_clause = f"(ra >= {ra_min} OR ra <= {ra_max - 360.0})"
        else:
            ra_clause = f"ra BETWEEN {ra_min} AND {ra_max}"
        query = (
            "SELECT TOP 20 ra, dec, "
            "mean_mag_g, median_mag_g, mean_cmag_g, median_cmag_g, "
            "mean_mag_r, median_mag_r, mean_cmag_r, median_cmag_r, "
            "mean_mag_i, median_mag_i, mean_cmag_i, median_cmag_i, "
            "mean_mag_z, median_mag_z, mean_cmag_z, median_cmag_z, "
            "mean_mag_y, median_mag_y, mean_cmag_y, median_cmag_y, "
            f"((ra-({ra}))*(ra-({ra})) + (dec-({dec}))*(dec-({dec}))) AS dist2 "
            "FROM decaps_dr2.object "
            f"WHERE {ra_clause} "
            f"AND dec BETWEEN {dec - radius_deg} AND {dec + radius_deg} "
            "ORDER BY dist2 ASC"
        )
        try:
            result = tap.search(query).to_table().to_pandas()
            if result.empty:
                statuses[cid] = "miss"
                continue
            candidate_matches: list[tuple[float, pd.Series]] = []
            for _, row in result.iterrows():
                match_ra = _safe_float(row.get("ra"))
                match_dec = _safe_float(row.get("dec"))
                if match_ra is None or match_dec is None:
                    continue
                wrapped_dra = ((match_ra - ra + 180.0) % 360.0) - 180.0
                mean_dec = 0.5 * (match_dec + dec)
                separation = math.hypot(
                    wrapped_dra * math.cos(math.radians(mean_dec)),
                    match_dec - dec,
                ) * 3600.0
                if separation <= float(radius_arcsec):
                    candidate_matches.append((float(separation), row))
            if not candidate_matches:
                statuses[cid] = "miss"
                continue
            candidate_matches.sort(key=lambda value: value[0])
            sep_arcsec, row = candidate_matches[0]
            values = {
                "g": (_row_float_value(row, ("mean_mag_g", "median_mag_g", "mean_cmag_g", "median_cmag_g")), None),
                "r": (_row_float_value(row, ("mean_mag_r", "median_mag_r", "mean_cmag_r", "median_cmag_r")), None),
                "i": (_row_float_value(row, ("mean_mag_i", "median_mag_i", "mean_cmag_i", "median_cmag_i")), None),
                "z": (_row_float_value(row, ("mean_mag_z", "median_mag_z", "mean_cmag_z", "median_cmag_z")), None),
                "Y": (_row_float_value(row, ("mean_mag_y", "median_mag_y", "mean_cmag_y", "median_cmag_y")), None),
            }
            flags = _counterpart_validation_flags(
                payload,
                row,
                source_key="decaps",
                separation_arcsec=sep_arcsec,
                candidate_separations_arcsec=[value[0] for value in candidate_matches],
                radius_arcsec=radius_arcsec,
            )
            candidate_rows = _rows_from_simple_mag_dict(
                cid,
                values,
                source="DECaPS",
                distance_pc=distance_pc_from_payload(payload),
                sep_arcsec=sep_arcsec,
                quality_flags=";".join(flags),
            )
            if candidate_rows:
                rows.extend(candidate_rows)
            else:
                statuses[cid] = "miss"
        except Exception as exc:
            statuses[cid] = "error"
            if progress_callback and idx == 1:
                progress_callback(f"[SED] decaps first lookup failed: {exc}")
            continue
        finally:
            _sleep_after_sed_request()
    return _fetch_result(rows, statuses)


def query_des_photometry(df: pd.DataFrame, progress_callback: ProgressCallback | None = None) -> pd.DataFrame:
    return query_vizier_source(df, "des", progress_callback=progress_callback)


def _nsc_band_name(value: object) -> str:
    text = _clean_text(value)
    if text.casefold() == "y":
        return "Y"
    if text.casefold() == "vr":
        return "VR"
    return text.casefold()


def _nsc_instrument_name(value: object) -> str:
    token = re.sub(r"[^a-z0-9]+", "", _clean_text(value).casefold())
    return NSC_INSTRUMENT_ALIASES.get(token, token)


def _nsc_object_mean_rows(
    row: pd.Series,
    *,
    candidate_id: str,
    object_id: str,
    distance_pc: float | None,
    sep_arcsec: float | None,
    extra_quality_flags: Iterable[str] = (),
) -> list[dict]:
    values = {
        "u": (_catalog_mag_value(row, "umag"), _catalog_mag_error(row, "uerr")),
        "g": (_catalog_mag_value(row, "gmag"), _catalog_mag_error(row, "gerr")),
        "r": (_catalog_mag_value(row, "rmag"), _catalog_mag_error(row, "rerr")),
        "i": (_catalog_mag_value(row, "imag"), _catalog_mag_error(row, "ierr")),
        "z": (_catalog_mag_value(row, "zmag"), _catalog_mag_error(row, "zerr")),
        "Y": (_catalog_mag_value(row, "ymag"), _catalog_mag_error(row, "yerr")),
        "VR": (_catalog_mag_value(row, "vrmag"), _catalog_mag_error(row, "vrerr")),
    }
    output = _rows_from_simple_mag_dict(
        candidate_id,
        values,
        source="NOIRLab NSC DR2",
        distance_pc=distance_pc,
        sep_arcsec=sep_arcsec,
        quality_flags=_catalog_quality_flags(
            row,
            ("flags", "class_star", "ndet"),
            ";".join(str(flag) for flag in extra_quality_flags if str(flag)),
        ),
    )
    identity_object = object_id or candidate_id
    for measurement in output:
        provenance = {
            "band": str(measurement.get("band") or ""),
            "kind": "nsc_dr2_object_mean",
            "object_id": object_id or None,
        }
        measurement.update(
            {
                "catalog_release": "DR2",
                "source_object_id": object_id or None,
                "catalog_measurement_id": (
                    f"nsc-dr2-object-mean:{identity_object}:{measurement.get('band')}"
                ),
                "correlation_group": f"nsc:{identity_object}:mixed",
                "provenance_json": json.dumps(
                    provenance, sort_keys=True, separators=(",", ":")
                ),
            }
        )
        try:
            from malca.review.sed_storage import make_sed_measurement_id

            measurement["measurement_id"] = make_sed_measurement_id(
                {
                    **measurement,
                    "catalog": "NOIRLab NSC DR2",
                    "release": "DR2",
                    "catalog_object_id": object_id or None,
                }
            )
        except (ImportError, TypeError, ValueError):
            pass
    return output


NSC_SEXTRACTOR_SEVERE_FLAG_MASK = 4 | 8 | 16 | 32 | 64 | 128
NSC_POINT_SOURCE_CLASS_STAR_MIN = 0.5


def _nsc_sextractor_flag(value: object) -> int | None:
    """Return a valid non-negative integer SExtractor flag, else ``None``."""
    number = _safe_float(value)
    if number is None or number < 0 or not float(number).is_integer():
        return None
    return int(number)


def _sorted_clean_values(values: Iterable[object]) -> list[str]:
    return sorted({_clean_text(value) for value in values if _clean_text(value)})


def _nsc_measurement_aggregate_rows(
    measurements: pd.DataFrame,
    *,
    candidate_id: str,
    object_id: str,
    distance_pc: float | None,
    sep_arcsec: float | None,
) -> list[dict]:
    """Collapse NSC measurements to one robust point per physical filter.

    One row per instrument+band avoids treating repeat visits as independent
    SED constraints.  The uncertainty retains the epoch-to-epoch robust
    scatter, which is important for MALCA's variable sources.
    """
    if measurements is None or measurements.empty:
        return []
    frame = pd.DataFrame(measurements).copy()
    frame["_instrument"] = frame.apply(
        lambda item: _nsc_instrument_name(
            _row_value(item, ("instrument", "exposure_instrument", "camera"))
        ),
        axis=1,
    )
    frame["_band"] = frame.apply(
        lambda item: _nsc_band_name(
            _row_value(item, ("meas_filter", "filter", "exposure_filter"))
        ),
        axis=1,
    )
    frame["_mag"] = frame.apply(
        lambda item: _catalog_mag_value(item, ("mag_auto", "mag")), axis=1
    )
    frame["_mag_err"] = frame.apply(
        lambda item: _catalog_mag_error(item, ("magerr_auto", "mag_err")), axis=1
    )
    frame["_sextractor_flags"] = frame.apply(
        lambda item: _nsc_sextractor_flag(
            _row_value(item, ("meas_flags", "flags"))
        ),
        axis=1,
    )
    frame["_class_star"] = frame.apply(
        lambda item: _safe_float(
            _row_value(item, ("meas_class_star", "class_star"))
        ),
        axis=1,
    )
    frame["_measid"] = frame.apply(
        lambda item: _clean_text(_row_value(item, ("measid", "measurement_id"))),
        axis=1,
    )
    frame["_exposure"] = frame.apply(
        lambda item: _clean_text(_row_value(item, ("exposure", "exposure_id"))),
        axis=1,
    )
    frame["_epoch"] = frame.apply(
        lambda item: _safe_float(_row_value(item, ("mjd", "exposure_mjd"))),
        axis=1,
    )
    frame = frame[
        frame["_instrument"].ne("")
        & frame["_band"].ne("")
    ].copy()
    if frame.empty:
        return []

    out: list[dict] = []
    for (instrument, band), group in frame.groupby(["_instrument", "_band"], sort=True):
        bp = bandpass_for("NOIRLab NSC DR2", str(band), instrument=str(instrument))
        if bp is None or bp.svo_filter_id is None:
            continue
        mag_values = pd.to_numeric(group["_mag"], errors="coerce")
        error_values = pd.to_numeric(group["_mag_err"], errors="coerce")
        severe_flags = group["_sextractor_flags"].map(
            lambda value: value is not None and bool(int(value) & NSC_SEXTRACTOR_SEVERE_FLAG_MASK)
        )
        accepted_mask = mag_values.notna() & error_values.notna() & ~severe_flags
        accepted = group.loc[accepted_mask].copy()
        if accepted.empty:
            continue
        magnitudes = pd.to_numeric(accepted["_mag"], errors="coerce").to_numpy(dtype=float)
        magnitudes = magnitudes[np.isfinite(magnitudes)]
        if magnitudes.size == 0:
            continue
        magnitude = float(np.median(magnitudes))
        reported_errors = pd.to_numeric(accepted["_mag_err"], errors="coerce").to_numpy(dtype=float)
        reported_errors = reported_errors[np.isfinite(reported_errors) & (reported_errors > 0)]
        reported_sem = (
            float(np.median(reported_errors)) / math.sqrt(float(magnitudes.size))
            if reported_errors.size
            else 0.0
        )
        robust_scatter = (
            1.4826 * float(np.median(np.abs(magnitudes - magnitude)))
            if magnitudes.size > 1
            else 0.0
        )
        magnitude_error = max(reported_sem, robust_scatter, 0.005)
        epoch_values = pd.to_numeric(accepted["_epoch"], errors="coerce").to_numpy(dtype=float)
        epoch_values = epoch_values[np.isfinite(epoch_values)]
        epoch_mjd = float(np.median(epoch_values)) if epoch_values.size else None
        class_values = pd.to_numeric(accepted["_class_star"], errors="coerce").to_numpy(dtype=float)
        class_values = class_values[np.isfinite(class_values) & (class_values >= 0) & (class_values <= 1)]
        class_star = float(np.median(class_values)) if class_values.size else None
        accepted_flags = [
            value for value in accepted["_sextractor_flags"].tolist() if value is not None
        ]
        missing_flag_count = int(accepted["_sextractor_flags"].isna().sum())
        mild_flag_values = sorted({int(value) for value in accepted_flags if int(value) & 3})
        diagnostic_reasons: list[str] = []
        if class_star is None:
            diagnostic_reasons.append("nsc_class_star_unassessed")
        elif class_star < NSC_POINT_SOURCE_CLASS_STAR_MIN:
            diagnostic_reasons.extend(
                (f"nsc_class_star_median={class_star:.3f}", "nsc_nonstellar_morphology", "bad_quality")
            )
        else:
            diagnostic_reasons.append(f"nsc_class_star_median={class_star:.3f}")
        if missing_flag_count:
            diagnostic_reasons.extend(
                (f"nsc_missing_flags_n={missing_flag_count}", "nsc_quality_unassessed", "bad_quality")
            )
        if mild_flag_values:
            diagnostic_reasons.extend(
                (
                    "nsc_blended_or_deblended",
                    "nsc_mild_flags=" + ",".join(str(value) for value in mild_flag_values),
                    "bad_quality",
                )
            )
        measurement_ids = _sorted_clean_values(group["_measid"])
        exposure_ids = _sorted_clean_values(group["_exposure"])
        accepted_measurement_ids = _sorted_clean_values(accepted["_measid"])
        accepted_exposure_ids = _sorted_clean_values(accepted["_exposure"])
        rejected_measurement_ids = sorted(set(measurement_ids) - set(accepted_measurement_ids))
        rejected_exposure_ids = sorted(set(exposure_ids) - set(accepted_exposure_ids))
        provenance = {
            "accepted_exposure_ids": accepted_exposure_ids,
            "accepted_measurement_ids": accepted_measurement_ids,
            "aggregate": "median_with_scatter_floor",
            "band": str(band),
            "epoch_mjd_max": float(np.max(epoch_values)) if epoch_values.size else None,
            "epoch_mjd_min": float(np.min(epoch_values)) if epoch_values.size else None,
            "exposure_ids": exposure_ids,
            "instrument": str(instrument),
            "measurement_ids": measurement_ids,
            "n_accepted": int(len(accepted)),
            "n_input": int(len(group)),
            "object_id": object_id,
            "rejected_exposure_ids": rejected_exposure_ids,
            "rejected_measurement_ids": rejected_measurement_ids,
            "sextractor_severe_flag_mask": NSC_SEXTRACTOR_SEVERE_FLAG_MASK,
        }
        provenance_json = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
        aggregate_hash = hashlib.sha256(provenance_json.encode("utf-8")).hexdigest()
        flags = [
            "nsc_measurement_level",
            f"nsc_instrument={instrument}",
            f"nsc_aggregate_n={len(accepted)}",
            f"nsc_rejected_quality_n={len(group) - len(accepted)}",
            *diagnostic_reasons,
        ]
        catalog_flags = sorted(
            {
                _clean_text(value)
                for value in accepted["_sextractor_flags"]
                if _clean_text(value)
            }
        )
        if catalog_flags:
            flags.append("flags=" + ",".join(catalog_flags))
        row = _row_from_bandpass(
            candidate_id=candidate_id,
            bandpass=bp,
            mag=magnitude,
            mag_err=magnitude_error,
            distance_pc=distance_pc,
            av=None,
            dereddened=False,
            sep_arcsec=sep_arcsec,
            quality_flags=";".join(flags),
        )
        if row is None:
            continue
        row.update(
            {
                "catalog_release": "DR2",
                "source_object_id": object_id,
                "catalog_measurement_id": f"nsc-dr2-aggregate:{aggregate_hash[:24]}",
                "instrument": instrument,
                "exposure_id": f"aggregate:{aggregate_hash[:24]}",
                "epoch_mjd": epoch_mjd,
                "correlation_group": f"nsc:{object_id}:{instrument}",
                "provenance_json": provenance_json,
                "native_value": magnitude,
                "native_error": magnitude_error,
                "native_unit": "mag",
                "observable_kind": "ab_mag",
                "passband_fidelity": (
                    "exact" if bp.response_kind == "instrument_response" else bp.response_kind
                ),
                "response_kind": bp.response_kind,
                "fit_policy": "diagnostic_only" if "bad_quality" in flags else bp.fit_policy,
            }
        )
        try:
            from malca.review.sed_storage import make_sed_measurement_id

            row["measurement_id"] = make_sed_measurement_id(
                {
                    **row,
                    "catalog": "NOIRLab NSC DR2",
                    "release": "DR2",
                    "catalog_object_id": object_id,
                }
            )
        except (ImportError, TypeError, ValueError):
            pass
        out.append(row)
    return out


def query_nsc_photometry(
    df: pd.DataFrame,
    radius_arcsec: float = 1.2,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Fetch exact NSC DR2 per-instrument photometry, with safe mean fallback."""
    rows: list[dict] = []
    statuses: dict[str, str] = _all_candidate_status(df, "miss")
    try:
        import pyvo
        tap = pyvo.dal.TAPService("https://datalab.noirlab.edu/tap")
    except Exception:
        return _fetch_result([], _all_candidate_status(df, "error"))

    if len(df) >= SED_BULK_XMATCH_MIN_CANDIDATES:
        matches, statuses = _bulk_tap_bbox_matches(
            tap,
            df,
            table="nsc_dr2.object",
            select_columns=(
                "id, ra, dec, umag, uerr, gmag, gerr, rmag, rerr, "
                "imag, ierr, zmag, zerr, ymag, yerr, vrmag, vrerr, "
                "flags, class_star, ndet"
            ),
            radius_arcsec=radius_arcsec,
            source_key="nsc",
            progress_callback=progress_callback,
        )
        payloads = _candidate_rows_by_id(df)
        for cid, candidate_matches in matches.items():
            payload_row = payloads.get(cid)
            if payload_row is None or not candidate_matches:
                continue
            sep_arcsec, selected = candidate_matches[0]
            object_id = _clean_text(_row_value(selected, ("objectid", "object_id", "id")))
            match_flags = _counterpart_validation_flags(
                payload_row.to_dict(),
                selected,
                source_key="nsc",
                separation_arcsec=sep_arcsec,
                candidate_separations_arcsec=[item[0] for item in candidate_matches],
                radius_arcsec=radius_arcsec,
            )
            candidate_rows = _nsc_object_mean_rows(
                selected,
                candidate_id=cid,
                object_id=object_id,
                distance_pc=distance_pc_from_payload(payload_row.to_dict()),
                sep_arcsec=sep_arcsec,
                extra_quality_flags=(
                    "nsc_bulk_object_mean",
                    "nsc_measurement_detail_deferred",
                    *match_flags,
                ),
            )
            if candidate_rows:
                rows.extend(candidate_rows)
        return _fetch_result(rows, statuses)

    total = len(df)
    radius_deg = float(radius_arcsec) / 3600.0
    for idx, (_, item) in enumerate(df.iterrows(), start=1):
        if progress_callback and (idx == 1 or idx % 500 == 0 or idx == total):
            progress_callback(f"[SED] nsc {idx}/{total}")
        cid = _candidate_id_for_row(item)
        ra, dec = _ra_dec_from_row(item)
        if ra is None or dec is None:
            statuses[cid] = "error"
            continue
        payload = item.to_dict()
        ra_pad = radius_deg / max(math.cos(math.radians(dec)), 0.01)
        ra_min = ra - ra_pad
        ra_max = ra + ra_pad
        if ra_min < 0:
            ra_clause = f"(ra >= {ra_min + 360.0} OR ra <= {ra_max})"
        elif ra_max >= 360.0:
            ra_clause = f"(ra >= {ra_min} OR ra <= {ra_max - 360.0})"
        else:
            ra_clause = f"ra BETWEEN {ra_min} AND {ra_max}"
        query = (
            "SELECT TOP 1000 id, ra, dec, "
            "umag, uerr, gmag, gerr, rmag, rerr, imag, ierr, "
            "zmag, zerr, ymag, yerr, vrmag, vrerr, flags, class_star, ndet "
            "FROM nsc_dr2.object "
            f"WHERE {ra_clause} "
            f"AND dec BETWEEN {dec - radius_deg} AND {dec + radius_deg}"
        )
        try:
            result = tap.search(query).to_table().to_pandas()
            if result.empty:
                statuses[cid] = "miss"
                continue
            nearest: tuple[float, pd.Series] | None = None
            for _, candidate_match in result.iterrows():
                match_ra = _safe_float(candidate_match.get("ra"))
                match_dec = _safe_float(candidate_match.get("dec"))
                if match_ra is None or match_dec is None:
                    continue
                wrapped_dra = ((match_ra - ra + 180.0) % 360.0) - 180.0
                mean_dec = 0.5 * (match_dec + dec)
                dra = wrapped_dra * math.cos(math.radians(mean_dec))
                ddec = match_dec - dec
                separation = math.hypot(dra, ddec) * 3600.0
                if nearest is None or separation < nearest[0]:
                    nearest = (separation, candidate_match)
            if nearest is None or nearest[0] > float(radius_arcsec):
                statuses[cid] = "miss"
                continue
            sep_arcsec, row = nearest
            object_id = _clean_text(_row_value(row, ("objectid", "object_id", "id")))
            exact_rows: list[dict] = []
            if object_id:
                escaped_object_id = object_id.replace("'", "''")
                measurement_query = (
                    "SELECT TOP 5000 m.measid, m.objectid, m.exposure, "
                    "m.filter AS meas_filter, m.mjd, m.mag_auto, m.magerr_auto, "
                    "m.flags AS meas_flags, m.class_star AS meas_class_star, "
                    "e.instrument, e.filter AS exposure_filter, e.mjd AS exposure_mjd "
                    "FROM nsc_dr2.meas AS m "
                    "JOIN nsc_dr2.exposure AS e ON m.exposure = e.exposure "
                    f"WHERE m.objectid = '{escaped_object_id}'"
                )
                try:
                    measurement_result = tap.search(measurement_query).to_table().to_pandas()
                    exact_rows = _nsc_measurement_aggregate_rows(
                        measurement_result,
                        candidate_id=cid,
                        object_id=object_id,
                        distance_pc=distance_pc_from_payload(payload),
                        sep_arcsec=sep_arcsec,
                    )
                except Exception as exc:
                    if progress_callback and idx == 1:
                        progress_callback(f"[SED] nsc measurement join unavailable; using object means: {exc}")
                finally:
                    _sleep_after_sed_request()
            object_mean_rows = _nsc_object_mean_rows(
                row,
                candidate_id=cid,
                object_id=object_id,
                distance_pc=distance_pc_from_payload(payload),
                sep_arcsec=sep_arcsec,
            )
            exact_keys = {
                (_clean_text(candidate.get("candidate_id")), _nsc_band_name(candidate.get("band")))
                for candidate in exact_rows
            }
            fallback_rows = [
                candidate
                for candidate in object_mean_rows
                if (_clean_text(candidate.get("candidate_id")), _nsc_band_name(candidate.get("band")))
                not in exact_keys
            ]
            candidate_rows = [*exact_rows, *fallback_rows]
            if candidate_rows:
                rows.extend(candidate_rows)
            else:
                statuses[cid] = "miss"
        except Exception as exc:
            statuses[cid] = "error"
            if progress_callback and idx == 1:
                progress_callback(f"[SED] nsc first lookup failed: {exc}")
            continue
        finally:
            _sleep_after_sed_request()
    return _set_fetch_statuses(pd.DataFrame(rows, columns=CANONICAL_SED_COLUMNS), statuses)


def _archive_query_position(
    row: pd.Series,
    *,
    epoch_jyear: float,
) -> tuple[float | None, float | None, str]:
    ra, dec = _ra_dec_from_row(row)
    if ra is None or dec is None:
        return None, None, "missing_coordinates"
    try:
        from astropy.time import Time
        from malca.enrichment.astrometry import propagate_linear_icrs

        propagated_ra, propagated_dec, method = propagate_linear_icrs(
            ra,
            dec,
            Time(float(epoch_jyear), format="jyear").mjd,
            pmra_mas_per_year=_safe_float(_row_value(row, ("pmra", "gaia_pmra"))),
            pmdec_mas_per_year=_safe_float(_row_value(row, ("pmdec", "gaia_pmdec"))),
            reference_epoch_jyear=(
                _safe_float(_row_value(row, ("ref_epoch", "gaia_ref_epoch"))) or 2016.0
            ),
        )
        return float(propagated_ra), float(propagated_dec), str(method)
    except Exception:
        return float(ra), float(dec), "static_propagation_unavailable"


def _nearest_archive_row(
    frame: pd.DataFrame,
    *,
    target_ra_deg: float,
    target_dec_deg: float,
    radius_arcsec: float,
) -> tuple[pd.Series | None, float | None, list[float]]:
    if frame is None or frame.empty:
        return None, None, []
    from malca.enrichment.astrometry import angular_separation_arcsec

    ra = pd.to_numeric(
        frame.apply(lambda row: _row_value(row, ("ra", "RA", "RAJ2000", "RA_ICRS")), axis=1),
        errors="coerce",
    )
    dec = pd.to_numeric(
        frame.apply(
            lambda row: _row_value(
                row,
                ("dec", "DEC", "DE", "DEJ2000", "DE_ICRS"),
            ),
            axis=1,
        ),
        errors="coerce",
    )
    valid = ra.notna() & dec.notna()
    if not valid.any():
        return None, None, []
    separations = pd.Series(
        angular_separation_arcsec(
            target_ra_deg,
            target_dec_deg,
            ra.loc[valid].to_numpy(dtype=float),
            dec.loc[valid].to_numpy(dtype=float),
        ),
        index=ra.loc[valid].index,
        dtype=float,
    )
    separations = separations[np.isfinite(separations) & (separations <= float(radius_arcsec))]
    if separations.empty:
        return None, None, []
    selected_index = separations.idxmin()
    return frame.loc[selected_index], float(separations.loc[selected_index]), sorted(
        float(value) for value in separations
    )


def _irsa_query_region_frame(
    row: pd.Series,
    *,
    catalog: str,
    epoch_jyear: float,
    radius_arcsec: float,
    columns: str = "*",
) -> tuple[pd.DataFrame, float | None, float | None, str]:
    from astroquery.ipac.irsa import Irsa
    from astropy.coordinates import SkyCoord

    ra, dec, coordinate_method = _archive_query_position(row, epoch_jyear=epoch_jyear)
    if ra is None or dec is None:
        return pd.DataFrame(), None, None, coordinate_method
    result = Irsa.query_region(
        SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
        catalog=catalog,
        radius=float(radius_arcsec) * u.arcsec,
        columns=columns,
    )
    if result is None:
        return pd.DataFrame(), ra, dec, coordinate_method
    if hasattr(result, "to_pandas"):
        frame = result.to_pandas()
    elif hasattr(result, "to_table"):
        table = result.to_table()
        frame = table.to_pandas() if hasattr(table, "to_pandas") else pd.DataFrame(table)
    else:
        frame = pd.DataFrame(result)
    return frame, ra, dec, coordinate_method


def _band_character(value: object, band_index: int) -> str:
    text = _clean_text(value)
    return text[band_index] if band_index < len(text) else ""


def query_irsa_allwise_photometry(
    df: pd.DataFrame,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Fetch canonical AllWISE W1-W4 profile-fit photometry directly from IRSA."""

    rows: list[dict] = []
    statuses = _all_candidate_status(df, "query_error")
    columns = ",".join(
        [
            "designation",
            "ra",
            "dec",
            "w1mpro",
            "w1sigmpro",
            "w2mpro",
            "w2sigmpro",
            "w3mpro",
            "w3sigmpro",
            "w4mpro",
            "w4sigmpro",
            "ph_qual",
            "cc_flags",
            "ext_flg",
            "nb",
            "na",
            "var_flg",
            "w1snr",
            "w2snr",
            "w3snr",
            "w4snr",
            "w1rchi2",
            "w2rchi2",
            "w3rchi2",
            "w4rchi2",
            "w1sat",
            "w2sat",
            "w3sat",
            "w4sat",
        ]
    )
    total = len(df)
    for idx, (_, item) in enumerate(df.iterrows(), start=1):
        cid = _candidate_id_for_row(item)
        if progress_callback and (idx == 1 or idx % 100 == 0 or idx == total):
            progress_callback(f"[SED] allwise {idx}/{total}")
        try:
            result, target_ra, target_dec, coordinate_method = _irsa_query_region_frame(
                item,
                catalog="allwise_p3as_psd",
                epoch_jyear=2010.5,
                radius_arcsec=3.0,
                columns=columns,
            )
            if target_ra is None or target_dec is None:
                statuses[cid] = "query_error"
                continue
            match, separation, separations = _nearest_archive_row(
                result,
                target_ra_deg=target_ra,
                target_dec_deg=target_dec,
                radius_arcsec=3.0,
            )
            if match is None:
                statuses[cid] = "covered_no_detection"
                continue
            designation = _clean_text(_row_value(match, ("designation", "source_id")))
            match_flags = _counterpart_validation_flags(
                item.to_dict(),
                match,
                source_key="allwise",
                separation_arcsec=separation,
                candidate_separations_arcsec=separations,
                radius_arcsec=3.0,
            )
            candidate_rows: list[dict] = []
            for band_index, band in enumerate(("W1", "W2", "W3", "W4")):
                number = band_index + 1
                mag = _catalog_mag_value(match, f"w{number}mpro")
                if mag is None:
                    continue
                mag_err = _catalog_mag_error(match, f"w{number}sigmpro")
                ph_qual = _band_character(_row_value(match, "ph_qual"), band_index).upper()
                cc_flag = _band_character(_row_value(match, "cc_flags"), band_index)
                snr = _row_float_value(match, f"w{number}snr")
                rchi2 = _row_float_value(match, f"w{number}rchi2")
                saturation = _row_float_value(match, f"w{number}sat")
                ext_flg = _row_float_value(match, "ext_flg")
                quality = [
                    *match_flags,
                    "irsa_direct",
                    f"ph_qual={ph_qual or 'unknown'}",
                    f"cc_flag={cc_flag or 'unknown'}",
                ]
                is_upper_limit = ph_qual == "U"
                bad = (
                    ph_qual not in {"A", "B", "U"}
                    or (cc_flag not in {"", "0"})
                    or (ext_flg is not None and ext_flg > 0)
                    or (snr is not None and snr < 2.0 and not is_upper_limit)
                    or (saturation is not None and saturation > 0)
                )
                if rchi2 is not None and rchi2 > 3.0:
                    quality.append("allwise_large_rchi2")
                if bad:
                    quality.append("bad_quality")
                bp = bandpass_for("AllWISE", band)
                if bp is None:
                    continue
                provenance = {
                    "catalog": "allwise_p3as_psd",
                    "catalog_object_id": designation or None,
                    "selected_sep_arcsec": separation,
                    "match_count_returned": len(separations),
                    "coordinate_method": coordinate_method,
                    "ph_qual": ph_qual or None,
                    "cc_flag": cc_flag or None,
                    "ext_flg": ext_flg,
                    "snr": snr,
                    "rchi2": rchi2,
                    "saturation": saturation,
                }
                sed_row = _row_from_bandpass(
                    candidate_id=cid,
                    bandpass=bp,
                    mag=mag,
                    mag_err=mag_err,
                    distance_pc=distance_pc_from_payload(item.to_dict()),
                    av=None,
                    dereddened=False,
                    sep_arcsec=separation,
                    quality_flags=";".join(dict.fromkeys(quality)),
                    is_upper_limit=is_upper_limit,
                    wavelength_metadata={
                        "catalog_release": "allwise_p3as_psd",
                        "source_object_id": designation or None,
                        "catalog_measurement_id": (
                            f"{designation}:{band}" if designation else None
                        ),
                        "instrument": "WISE",
                        "epoch_mjd": 55379.0,
                        "correlation_group": (
                            f"allwise:{designation}" if designation else f"allwise:{cid}"
                        ),
                        "provenance_json": _canonical_sed_json_text(provenance),
                    },
                    policy_payload=item.to_dict(),
                )
                if sed_row is not None:
                    if bad:
                        sed_row["fit_policy"] = "diagnostic_only"
                    candidate_rows.append(sed_row)
            rows.extend(candidate_rows)
            statuses[cid] = "catalog_detection" if candidate_rows else "covered_no_detection"
        except Exception as exc:
            statuses[cid] = "query_error"
            if progress_callback and idx == 1:
                progress_callback(f"[SED] allwise first lookup failed: {exc}")
        finally:
            _sleep_after_sed_request()
    return _fetch_result(rows, statuses)


def query_irsa_spitzer_photometry(
    df: pd.DataFrame,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Fetch SEIP Source List photometry directly from IRSA without fallbacks."""

    rows: list[dict] = []
    statuses = _all_candidate_status(df, "query_error")
    total = len(df)
    band_specs = {
        "IRAC1": ("i1", "i1_f_ap1", "i1_df_ap1", "Spitzer/IRAC.I1"),
        "IRAC2": ("i2", "i2_f_ap1", "i2_df_ap1", "Spitzer/IRAC.I2"),
        "IRAC3": ("i3", "i3_f_ap1", "i3_df_ap1", "Spitzer/IRAC.I3"),
        "IRAC4": ("i4", "i4_f_ap1", "i4_df_ap1", "Spitzer/IRAC.I4"),
        "MIPS24": ("m1", "m1_f_psf", "m1_df_psf", "Spitzer/MIPS.24mu"),
    }
    for idx, (_, item) in enumerate(df.iterrows(), start=1):
        cid = _candidate_id_for_row(item)
        if progress_callback and (idx == 1 or idx % 100 == 0 or idx == total):
            progress_callback(f"[SED] spitzer {idx}/{total}")
        try:
            result, target_ra, target_dec, coordinate_method = _irsa_query_region_frame(
                item,
                catalog="slphotdr4",
                epoch_jyear=2008.0,
                radius_arcsec=3.0,
            )
            if target_ra is None or target_dec is None:
                statuses[cid] = "query_error"
                continue
            match, separation, separations = _nearest_archive_row(
                result,
                target_ra_deg=target_ra,
                target_dec_deg=target_dec,
                radius_arcsec=3.0,
            )
            if match is None:
                statuses[cid] = "catalog_no_match"
                continue
            object_id = _clean_text(_row_value(match, ("objid", "source_id", "id")))
            match_flags = _counterpart_validation_flags(
                item.to_dict(),
                match,
                source_key="spitzer",
                separation_arcsec=separation,
                candidate_separations_arcsec=separations,
                radius_arcsec=3.0,
            )
            candidate_rows: list[dict] = []
            for band, (prefix, flux_col, error_col, _filter_id) in band_specs.items():
                raw_flux = _row_float_value(match, flux_col)
                if raw_flux is None or raw_flux <= 0:
                    continue
                raw_error = _row_float_value(match, error_col)
                flux_type = _row_float_value(match, f"{prefix}_fluxtype")
                flux_flag = _row_float_value(match, f"{prefix}_fluxflag")
                soft_sat = _row_float_value(match, f"{prefix}_softsatflag")
                brt_frac = _row_float_value(match, f"{prefix}_brtfrac")
                ext_frac = _row_float_value(match, f"{prefix}_extfrac")
                snr = _row_float_value(match, f"{prefix}_snr")
                quality = [
                    *match_flags,
                    "irsa_direct",
                    f"fluxtype={flux_type if flux_type is not None else 'unknown'}",
                ]
                bad = flux_type != 1
                if band.startswith("IRAC"):
                    if flux_flag != 0:
                        quality.append("spitzer_flux_region_flag")
                        bad = True
                    if soft_sat != 0:
                        quality.append("spitzer_soft_saturation")
                        bad = True
                else:
                    if brt_frac is not None and brt_frac >= 0.5:
                        quality.append("spitzer_bright_region")
                        bad = True
                    if ext_frac is not None and ext_frac >= 0.5:
                        quality.append("spitzer_extended_region")
                        bad = True
                if bad:
                    quality.append("bad_quality")
                bp = bandpass_for("Spitzer SEIP", band)
                if bp is None:
                    continue
                provenance = {
                    "catalog": "slphotdr4",
                    "catalog_object_id": object_id or None,
                    "selected_sep_arcsec": separation,
                    "match_count_returned": len(separations),
                    "coordinate_method": coordinate_method,
                    "fluxtype": flux_type,
                    "fluxflag": flux_flag,
                    "softsatflag": soft_sat,
                    "bright_fraction": brt_frac,
                    "extended_fraction": ext_frac,
                    "snr": snr,
                }
                sed_row = _row_from_bandpass(
                    candidate_id=cid,
                    bandpass=bp,
                    mag=None,
                    mag_err=None,
                    distance_pc=distance_pc_from_payload(item.to_dict()),
                    av=None,
                    dereddened=False,
                    sep_arcsec=separation,
                    quality_flags=";".join(dict.fromkeys(quality)),
                    flux_nu_jy=raw_flux * 1.0e-6,
                    flux_nu_jy_err=(
                        raw_error * 1.0e-6
                        if raw_error is not None and raw_error > 0
                        else None
                    ),
                    wavelength_metadata={
                        "catalog_release": "slphotdr4",
                        "source_object_id": object_id or None,
                        "catalog_measurement_id": (
                            f"{object_id}:{band}" if object_id else None
                        ),
                        "instrument": "IRAC" if band.startswith("IRAC") else "MIPS",
                        "correlation_group": (
                            f"spitzer-seip:{object_id}" if object_id else f"spitzer-seip:{cid}"
                        ),
                        "provenance_json": _canonical_sed_json_text(provenance),
                    },
                    policy_payload=item.to_dict(),
                )
                if sed_row is not None:
                    sed_row["fit_policy"] = "diagnostic_only"
                    candidate_rows.append(sed_row)
            rows.extend(candidate_rows)
            statuses[cid] = "catalog_detection" if candidate_rows else "catalog_no_match"
        except Exception as exc:
            statuses[cid] = "query_error"
            if progress_callback and idx == 1:
                progress_callback(f"[SED] spitzer first lookup failed: {exc}")
        finally:
            _sleep_after_sed_request()
    return _fetch_result(rows, statuses)


@dataclass(frozen=True)
class VizierFluxSpec:
    source: str
    catalog: str
    radius_arcsec: float
    bands: dict[str, tuple[str | tuple[str, ...], str | tuple[str, ...] | None]]
    flux_scale_to_jy: float = 1.0
    error_is_percent: bool = False


VIZIER_FLUX_SPECS: dict[str, VizierFluxSpec] = {
    "akari": VizierFluxSpec("AKARI", "II/297/irc", 5.0, {
        "S9W": ("S09", "e_S09"),
        "L18W": ("S18", "e_S18"),
    }),
    "akari_fis": VizierFluxSpec("AKARI", "II/298/fis", 20.0, {
        "N60": (("S65", "F65", "N60", "Flux65"), ("e_S65", "e_F65", "e_N60")),
        "WIDE-S": (("S90", "F90", "WIDES", "Flux90"), ("e_S90", "e_F90", "e_WIDES")),
        "WIDE-L": (("S140", "F140", "WIDEL", "Flux140"), ("e_S140", "e_F140", "e_WIDEL")),
        "N160": (("S160", "F160", "N160", "Flux160"), ("e_S160", "e_F160", "e_N160")),
    }),
    "iras": VizierFluxSpec("IRAS", "II/125/main", 30.0, {
        "12": (("Fnu_12", "F12", "f12"), ("e_Fnu_12", "e_F12")),
        "25": (("Fnu_25", "F25", "f25"), ("e_Fnu_25", "e_F25")),
        "60": (("Fnu_60", "F60", "f60"), ("e_Fnu_60", "e_F60")),
        "100": (("Fnu_100", "F100", "f100"), ("e_Fnu_100", "e_F100")),
    }, error_is_percent=True),
    "herschel70": VizierFluxSpec("Herschel", "VIII/106/hppsc070", 8.0, {
        "PACS70": (("Flux", "flux", "F70", "Fnu"), ("snrnoise", "rms", "e_Flux", "e_flux", "e_F70")),
    }, flux_scale_to_jy=1.0e-3),
    "herschel100": VizierFluxSpec("Herschel", "VIII/106/hppsc100", 10.0, {
        "PACS100": (("Flux", "flux", "F100", "Fnu"), ("snrnoise", "rms", "e_Flux", "e_flux", "e_F100")),
    }, flux_scale_to_jy=1.0e-3),
    "herschel160": VizierFluxSpec("Herschel", "VIII/106/hppsc160", 14.0, {
        "PACS160": (("Flux", "flux", "F160", "Fnu"), ("snrnoise", "rms", "e_Flux", "e_flux", "e_F160")),
    }, flux_scale_to_jy=1.0e-3),
}


_FLUX_CATALOG_COORDINATE_ALIASES = (
    ("RAJ2000", "RA_ICRS", "RAICRS", "RAdeg", "RA"),
    ("DEJ2000", "DE_ICRS", "DEICRS", "DEdeg", "DEC", "Dec"),
)


def _catalog_scalar(row: pd.Series, aliases: str | Iterable[str]) -> object | None:
    """Return one non-missing catalog value without stringifying sentinels."""
    value = _row_value(row, aliases)
    if value is None:
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _catalog_identifier(row: pd.Series) -> str:
    value = _catalog_scalar(
        row,
        (
            "Name",
            "name",
            "Source",
            "source",
            "SSTSL2",
            "objName",
            "AKARI",
            "IRAS",
            "Object",
            "ID",
        ),
    )
    text = _clean_text(value)
    if text:
        return text
    ra = _row_float_value(row, _FLUX_CATALOG_COORDINATE_ALIASES[0])
    dec = _row_float_value(row, _FLUX_CATALOG_COORDINATE_ALIASES[1])
    if ra is not None and dec is not None:
        return f"coord:{ra:.8f},{dec:.8f}"
    return ""


def _flux_catalog_separations_arcsec(
    result: pd.DataFrame,
    target: object,
) -> pd.Series:
    """Return VizieR cone-match separations, with a spherical fallback."""
    separations = pd.Series(np.nan, index=result.index, dtype=float)
    distance_column = next(
        (column for column in result.columns if str(column).strip().casefold() == "_r"),
        None,
    )
    if distance_column is not None:
        separations = pd.to_numeric(result[distance_column], errors="coerce")

    missing = ~np.isfinite(separations.to_numpy(dtype=float))
    if np.any(missing):
        try:
            from astropy.coordinates import SkyCoord

            for row_index in result.index[missing]:
                row = result.loc[row_index]
                row_ra = _row_float_value(row, _FLUX_CATALOG_COORDINATE_ALIASES[0])
                row_dec = _row_float_value(row, _FLUX_CATALOG_COORDINATE_ALIASES[1])
                if row_ra is None or row_dec is None:
                    continue
                counterpart = SkyCoord(ra=row_ra * u.deg, dec=row_dec * u.deg)
                separations.loc[row_index] = float(target.separation(counterpart).arcsec)
        except Exception:
            pass
    return separations


def _hex_band_flag(value: object, band_index: int) -> bool:
    if value is None:
        return False
    try:
        if isinstance(value, (int, np.integer)):
            encoded = int(value)
        elif isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
            encoded = int(value)
        else:
            encoded = int(str(value).strip(), 16)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(encoded & (1 << int(band_index)))


def _truthy_catalog_flag(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().casefold() not in {"", "0", "false", "f", "no", "n", "none", "nan", "-"}
    try:
        return bool(float(value)) if np.isfinite(float(value)) else False
    except (TypeError, ValueError):
        return bool(value)


def _flux_catalog_quality(
    spec_key: str,
    band: str,
    row: pd.Series,
) -> tuple[list[str], bool, dict[str, object]]:
    """Interpret only documented band-specific direct-flux quality fields."""
    flags: list[str] = []
    is_upper_limit = False
    details: dict[str, object] = {}

    if spec_key in {"akari", "akari_fis"}:
        suffix = {
            "S9W": "S09",
            "L18W": "S18",
            "N60": "S65",
            "WIDE-S": "S90",
            "WIDE-L": "S140",
            "N160": "S160",
        }.get(str(band))
        quality = _row_float_value(row, (f"q_{suffix}", f"q{suffix}")) if suffix else None
        if quality is not None:
            quality_int = int(round(quality))
            details["flux_quality"] = quality_int
            if quality_int == 2:
                flags.extend(("akari_flux_unreliable", "bad_quality"))
            elif quality_int == 1:
                flags.extend(("akari_source_unconfirmed", "bad_quality"))
            elif quality_int <= 0:
                flags.extend(("akari_unobserved", "bad_quality"))
            elif quality_int != 3:
                flags.extend(("akari_quality_unknown", "bad_quality"))
        else:
            flags.append("akari_quality_unassessed")
        if suffix:
            raw_flag = _catalog_scalar(row, (f"f_{suffix}", f"f{suffix}"))
            if raw_flag is not None:
                details["data_quality_flag"] = raw_flag

    elif spec_key == "iras":
        quality = _row_float_value(row, (f"q_Fnu_{band}", f"q_F{band}", f"q{band}"))
        if quality is not None:
            quality_int = int(round(quality))
            details["flux_quality"] = quality_int
            if quality_int == 1:
                is_upper_limit = True
                flags.append("iras_upper_limit")
            elif quality_int == 2:
                flags.append("iras_moderate_quality")
            elif quality_int != 3:
                flags.extend(("iras_quality_unknown", "bad_quality"))
        else:
            flags.append("iras_quality_unassessed")
        band_index = {"12": 0, "25": 1, "60": 2, "100": 3}.get(str(band), 0)
        for column, flag_name in (
            ("Disc", "iras_discrepant_flux"),
            ("Confuse", "iras_confused_flux"),
            ("HSDFlag", "iras_high_source_density"),
        ):
            raw_value = _catalog_scalar(row, column)
            if raw_value is not None:
                details[column] = raw_value
            if _hex_band_flag(raw_value, band_index):
                flags.extend((flag_name, "bad_quality"))
        for prefix, flag_name in (
            ("SES1", "iras_nearby_seconds_confirmed_extension"),
            ("SES2", "iras_nearby_weeks_confirmed_extension"),
        ):
            raw_value = _catalog_scalar(row, f"{prefix}_{band}")
            if raw_value is not None:
                details[f"{prefix}_{band}"] = raw_value
            if _truthy_catalog_flag(raw_value):
                flags.extend((flag_name, "bad_quality"))
        correlation = _catalog_scalar(row, f"CC_{band}")
        if correlation is not None:
            details[f"CC_{band}"] = correlation

    elif spec_key.startswith("herschel"):
        for column, flag_name in (
            ("Edge", "herschel_edge"),
            ("Blend", "herschel_blended"),
            ("Warmat", "herschel_warm_attitude"),
            ("SSOmap", "herschel_sso_map"),
        ):
            raw_value = _catalog_scalar(row, column)
            if raw_value is not None:
                details[column] = raw_value
            if _truthy_catalog_flag(raw_value):
                flags.extend((flag_name, "bad_quality"))

    return flags, is_upper_limit, details


def _rows_from_flux_catalog_match(
    *,
    candidate_id: str,
    payload: Mapping[str, object],
    spec_key: str,
    spec: VizierFluxSpec,
    row: pd.Series,
    separation_arcsec: float | None,
    candidate_separations_arcsec: Iterable[float],
    result_count: int,
    row_limit_reached: bool = False,
) -> list[dict]:
    separations = sorted(
        float(value)
        for value in candidate_separations_arcsec
        if value is not None and np.isfinite(float(value))
    )
    second_nearest_sep = separations[1] if len(separations) > 1 else None
    source_object_id = _catalog_identifier(row)
    obs_id = _clean_text(_catalog_scalar(row, ("ObsId", "obsid", "OBSID")))
    match_flags = ["confusion_risk", "flux_catalog"]
    match_flags.extend(
        _counterpart_validation_flags(
            payload,
            row,
            source_key=spec_key,
            separation_arcsec=separation_arcsec,
            candidate_separations_arcsec=separations,
            radius_arcsec=spec.radius_arcsec,
        )
    )
    if row_limit_reached:
        match_flags.append("catalog_match_row_limit_reached")

    output: list[dict] = []
    for band, (flux_aliases, err_aliases) in spec.bands.items():
        raw_flux = _row_float_value(row, flux_aliases)
        if raw_flux is None or raw_flux <= 0:
            continue
        flux = raw_flux * float(spec.flux_scale_to_jy)
        bp = bandpass_for(spec.source, band)
        if bp is None:
            continue
        raw_flux_err = _row_float_value(row, err_aliases)
        if spec.error_is_percent:
            flux_err = (
                flux * raw_flux_err / 100.0
                if raw_flux_err is not None and raw_flux_err > 0
                else None
            )
        else:
            flux_err = (
                raw_flux_err * float(spec.flux_scale_to_jy)
                if raw_flux_err is not None and raw_flux_err > 0
                else None
            )
        quality_flags, is_upper_limit, quality_details = _flux_catalog_quality(
            spec_key,
            band,
            row,
        )
        if flux_err is not None and flux_err / flux >= 0.3:
            quality_flags.extend(("low_snr_flux", "bad_quality"))
        provenance = {
            "catalog": spec.catalog,
            "catalog_object_id": source_object_id or None,
            "selected_sep_arcsec": separation_arcsec,
            "match_count_returned": int(result_count),
            "second_nearest_sep_arcsec": second_nearest_sep,
            "row_limit": 5,
            "quality": quality_details,
        }
        wavelength_metadata = {
            "catalog_release": spec.catalog,
            "source_object_id": source_object_id or None,
            "catalog_measurement_id": (
                f"{source_object_id}:{obs_id}:{band}"
                if source_object_id and obs_id
                else f"{source_object_id}:{band}"
                if source_object_id
                else None
            ),
            "exposure_id": obs_id or None,
            "provenance_json": _canonical_sed_json_text(provenance),
        }
        sed_row = _row_from_bandpass(
            candidate_id=candidate_id,
            bandpass=bp,
            mag=None,
            mag_err=None,
            distance_pc=distance_pc_from_payload(payload),
            av=None,
            dereddened=False,
            sep_arcsec=separation_arcsec,
            quality_flags=";".join([*dict.fromkeys(match_flags), *quality_flags]),
            is_upper_limit=is_upper_limit,
            flux_nu_jy=flux,
            flux_nu_jy_err=flux_err,
            wavelength_metadata=wavelength_metadata,
            policy_payload=payload,
        )
        if sed_row is not None:
            if "bad_quality" in quality_flags or "bad_quality" in match_flags:
                sed_row["fit_policy"] = "diagnostic_only"
            output.append(sed_row)
    return output


def _flux_measurement_schema_present(row: pd.Series, spec: VizierFluxSpec) -> bool:
    return any(
        _row_has_any_column(row, flux_aliases)
        for flux_aliases, _error_aliases in spec.bands.values()
    )


def query_flux_catalog_source(
    df: pd.DataFrame,
    source_key: str,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Fetch a flux-density catalog source from VizieR, best effort."""
    requested_keys = {
        "akari": ("akari", "akari_fis"),
        "iras": ("iras",),
        "herschel": ("herschel70", "herschel100", "herschel160"),
    }.get(source_key, (source_key,))
    out: list[dict] = []
    statuses: dict[str, str] = _all_candidate_status(df, "miss")

    if len(df) >= SED_BULK_XMATCH_MIN_CANDIDATES:
        payloads = _candidate_rows_by_id(df)
        for key in requested_keys:
            spec = VIZIER_FLUX_SPECS.get(key)
            if spec is None:
                continue
            matches, catalog_statuses = _bulk_vizier_matches(
                df,
                catalog=spec.catalog,
                radius_arcsec=spec.radius_arcsec,
                source_key=f"{source_key}:{key}",
                progress_callback=progress_callback,
            )
            for cid, status_value in catalog_statuses.items():
                if status_value in SED_CACHE_RETRYABLE_STATUSES:
                    statuses[cid] = status_value
            for cid, candidate_matches in matches.items():
                payload_row = payloads.get(cid)
                if payload_row is None or not candidate_matches:
                    continue
                separation, selected = candidate_matches[0]
                candidate_rows = _rows_from_flux_catalog_match(
                    candidate_id=cid,
                    payload=payload_row.to_dict(),
                    spec_key=key,
                    spec=spec,
                    row=selected,
                    separation_arcsec=separation,
                    candidate_separations_arcsec=[item[0] for item in candidate_matches],
                    result_count=len(candidate_matches),
                )
                if not candidate_rows and not _flux_measurement_schema_present(selected, spec):
                    statuses[cid] = "error"
                out.extend(candidate_rows)
        return _fetch_result(out, statuses)

    try:
        from astropy.coordinates import SkyCoord
        from astroquery.vizier import Vizier
    except Exception:
        return _fetch_result([], _all_candidate_status(df, "error"))

    # ``+_r`` asks VizieR to sort before applying ``row_limit``.  We still
    # select the numeric minimum locally so correctness does not depend on
    # service ordering or a mocked response.
    viz = Vizier(columns=["**", "+_r"], row_limit=5)
    viz.TIMEOUT = VIZIER_QUERY_TIMEOUT_SEC
    total = len(df)
    for idx, (_, item) in enumerate(df.iterrows(), start=1):
        if progress_callback and (idx == 1 or idx % 500 == 0 or idx == total):
            progress_callback(f"[SED] {source_key} {idx}/{total}")
        cid = _candidate_id_for_row(item)
        ra, dec = _ra_dec_from_row(item)
        if ra is None or dec is None:
            statuses[cid] = "error"
            continue
        statuses[cid] = "miss"
        payload = item.to_dict()
        for key in requested_keys:
            spec = VIZIER_FLUX_SPECS.get(key)
            if spec is None:
                continue
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
                separations = _flux_catalog_separations_arcsec(result, target)
                finite_separations = separations[np.isfinite(separations)]
                if not finite_separations.empty:
                    within_radius = finite_separations[finite_separations <= float(spec.radius_arcsec)]
                    if within_radius.empty:
                        continue
                    selected_index = within_radius.idxmin()
                    ordered_separations = np.sort(within_radius.to_numpy(dtype=float))
                    sep_arcsec = float(within_radius.loc[selected_index])
                    second_nearest_sep = (
                        float(ordered_separations[1]) if ordered_separations.size > 1 else None
                    )
                    match_count = int(len(within_radius))
                else:
                    # Preserve compatibility with single-row catalog replies
                    # that omit both ``_r`` and coordinates, but make the loss
                    # of astrometric provenance explicit.
                    selected_index = result.index[0]
                    sep_arcsec = None
                    ordered_separations = np.asarray([], dtype=float)
                    match_count = int(len(result))
                row = result.loc[selected_index]
                candidate_rows = _rows_from_flux_catalog_match(
                    candidate_id=cid,
                    payload=payload,
                    spec_key=key,
                    spec=spec,
                    row=row,
                    separation_arcsec=sep_arcsec,
                    candidate_separations_arcsec=ordered_separations,
                    result_count=match_count,
                    row_limit_reached=len(result) >= 5,
                )
                if not candidate_rows and not _flux_measurement_schema_present(row, spec):
                    statuses[cid] = "error"
                out.extend(candidate_rows)
            except Exception as exc:
                statuses[cid] = "error"
                if progress_callback and idx == 1:
                    progress_callback(f"[SED] {source_key} first lookup failed: {exc}")
                continue
            finally:
                _sleep_after_sed_request()
    return _fetch_result(out, statuses)


def _direct_flux_sed_row(
    *,
    candidate_id: str,
    payload: Mapping[str, object],
    source: str,
    band: str,
    flux_jy: float,
    flux_error_jy: float | None,
    separation_arcsec: float | None,
    catalog_release: str,
    source_object_id: str | None,
    observation_id: str | None,
    instrument: str,
    quality_flags: Iterable[str],
    provenance: Mapping[str, object],
) -> dict | None:
    bp = bandpass_for(source, band)
    if bp is None or not np.isfinite(float(flux_jy)) or float(flux_jy) <= 0:
        return None
    flags = list(dict.fromkeys(str(value) for value in quality_flags if str(value)))
    row = _row_from_bandpass(
        candidate_id=candidate_id,
        bandpass=bp,
        mag=None,
        mag_err=None,
        distance_pc=distance_pc_from_payload(payload),
        av=None,
        dereddened=False,
        sep_arcsec=separation_arcsec,
        quality_flags=";".join(flags),
        flux_nu_jy=float(flux_jy),
        flux_nu_jy_err=(
            float(flux_error_jy)
            if flux_error_jy is not None and np.isfinite(float(flux_error_jy))
            else None
        ),
        wavelength_metadata={
            "catalog_release": catalog_release,
            "source_object_id": source_object_id,
            "catalog_measurement_id": (
                f"{source_object_id}:{observation_id}:{band}"
                if source_object_id and observation_id
                else f"{source_object_id}:{band}"
                if source_object_id
                else None
            ),
            "instrument": instrument,
            "exposure_id": observation_id,
            "correlation_group": (
                f"{catalog_release}:{observation_id or source_object_id or candidate_id}"
            ),
            "provenance_json": _canonical_sed_json_text(dict(provenance)),
        },
        policy_payload=payload,
    )
    if row is not None:
        row["fit_policy"] = "diagnostic_only"
    return row


def query_herschel_irsa_hsa_photometry(
    df: pd.DataFrame,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Fetch HPPSC2 and SPIRE source catalogs from their primary archives."""

    rows: list[dict] = []
    statuses = _all_candidate_status(df, "query_error")
    pacs_specs = (
        ("hppsc2_bs", "PACS70", "f70", "e_f70", 8.0),
        ("hppsc2_bl", "PACS100", "f100", "e_f100", 10.0),
        ("hppsc2_r", "PACS160", "f160", "e_f160", 14.0),
    )
    spire_specs = (
        ("hsa.spire_point_source_250", "SPIRE250", 18.0),
        ("hsa.spire_point_source_350", "SPIRE350", 25.0),
        ("hsa.spire_point_source_500", "SPIRE500", 36.0),
    )
    total = len(df)
    for idx, (_, item) in enumerate(df.iterrows(), start=1):
        cid = _candidate_id_for_row(item)
        if progress_callback and (idx == 1 or idx % 50 == 0 or idx == total):
            progress_callback(f"[SED] herschel {idx}/{total}")
        target_ra, target_dec, coordinate_method = _archive_query_position(
            item,
            epoch_jyear=2011.0,
        )
        if target_ra is None or target_dec is None:
            statuses[cid] = "query_error"
            continue
        candidate_rows: list[dict] = []
        query_failed = False
        for catalog, band, flux_column, error_column, radius in pacs_specs:
            try:
                result, _ra, _dec, _method = _irsa_query_region_frame(
                    item,
                    catalog=catalog,
                    epoch_jyear=2011.0,
                    radius_arcsec=radius,
                )
                match, separation, separations = _nearest_archive_row(
                    result,
                    target_ra_deg=target_ra,
                    target_dec_deg=target_dec,
                    radius_arcsec=radius,
                )
                if match is None:
                    continue
                raw_flux_mjy = _row_float_value(match, flux_column)
                if raw_flux_mjy is None or raw_flux_mjy <= 0:
                    continue
                raw_error_mjy = _row_float_value(match, error_column)
                source_id = _clean_text(_row_value(match, ("cntr", "source_id")))
                obs_id = _clean_text(_row_value(match, ("obsid", "OBSID")))
                snr = _row_float_value(match, ("s_n", "snr"))
                structure_noise = _row_float_value(match, "strn")
                quality = ["confusion_risk", "irsa_direct", "hppsc2_high_reliability"]
                if len(separations) > 1:
                    quality.append("multiple_catalog_matches")
                if snr is not None and snr < 3.0:
                    quality.extend(("herschel_low_snr", "bad_quality"))
                provenance = {
                    "catalog": catalog,
                    "catalog_object_id": source_id or None,
                    "observation_id": obs_id or None,
                    "selected_sep_arcsec": separation,
                    "match_count_returned": len(separations),
                    "coordinate_method": coordinate_method,
                    "snr": snr,
                    "structure_noise_mjy_sr": structure_noise,
                }
                sed_row = _direct_flux_sed_row(
                    candidate_id=cid,
                    payload=item.to_dict(),
                    source="Herschel",
                    band=band,
                    flux_jy=raw_flux_mjy * 1.0e-3,
                    flux_error_jy=(
                        raw_error_mjy * 1.0e-3
                        if raw_error_mjy is not None and raw_error_mjy > 0
                        else None
                    ),
                    separation_arcsec=separation,
                    catalog_release=catalog,
                    source_object_id=source_id or None,
                    observation_id=obs_id or None,
                    instrument="PACS",
                    quality_flags=quality,
                    provenance=provenance,
                )
                if sed_row is not None:
                    candidate_rows.append(sed_row)
            except Exception as exc:
                query_failed = True
                if progress_callback and idx == 1:
                    progress_callback(f"[SED] {catalog} first lookup failed: {exc}")
            finally:
                _sleep_after_sed_request()

        try:
            from astroquery.esa.hsa import HSA

            for table_name, band, radius in spire_specs:
                radius_deg = float(radius) / 3600.0
                query = (
                    f"SELECT TOP 5 * FROM {table_name} "
                    "WHERE 1=CONTAINS("
                    "POINT('ICRS', ra, dec), "
                    f"CIRCLE('ICRS', {target_ra:.10f}, {target_dec:.10f}, {radius_deg:.10f})"
                    ")"
                )
                try:
                    table = HSA.query_hsa_tap(query)
                    result = (
                        table.to_pandas()
                        if hasattr(table, "to_pandas")
                        else pd.DataFrame(table)
                    )
                    match, separation, separations = _nearest_archive_row(
                        result,
                        target_ra_deg=target_ra,
                        target_dec_deg=target_dec,
                        radius_arcsec=radius,
                    )
                    if match is None:
                        continue
                    raw_flux_mjy = _row_float_value(match, "flux")
                    if raw_flux_mjy is None or raw_flux_mjy <= 0:
                        continue
                    raw_error_mjy = _row_float_value(match, ("flux_err", "fluxtml_err"))
                    source_id = _clean_text(_row_value(match, "source_id"))
                    snr = _row_float_value(match, "snr")
                    quality = ["confusion_risk", "hsa_direct", "spire_point_source_catalog"]
                    bad_flag_columns = (
                        "insterr_flag",
                        "extsrc_flag",
                        "largegal_flag",
                        "mapedge_flag",
                        "ssocont_flag",
                    )
                    bad = False
                    quality_payload: dict[str, object] = {}
                    for column in bad_flag_columns:
                        value = _catalog_scalar(match, column)
                        if value is None:
                            continue
                        quality_payload[column] = value
                        if _truthy_catalog_flag(value):
                            quality.append(f"herschel_{column}")
                            bad = True
                    if snr is not None and snr < 3.0:
                        quality.append("herschel_low_snr")
                        bad = True
                    if bad:
                        quality.append("bad_quality")
                    provenance = {
                        "catalog": table_name,
                        "catalog_object_id": source_id or None,
                        "selected_sep_arcsec": separation,
                        "match_count_returned": len(separations),
                        "coordinate_method": coordinate_method,
                        "snr": snr,
                        "quality": quality_payload,
                    }
                    sed_row = _direct_flux_sed_row(
                        candidate_id=cid,
                        payload=item.to_dict(),
                        source="Herschel",
                        band=band,
                        flux_jy=raw_flux_mjy * 1.0e-3,
                        flux_error_jy=(
                            raw_error_mjy * 1.0e-3
                            if raw_error_mjy is not None and raw_error_mjy > 0
                            else None
                        ),
                        separation_arcsec=separation,
                        catalog_release=table_name,
                        source_object_id=source_id or None,
                        observation_id=None,
                        instrument="SPIRE",
                        quality_flags=quality,
                        provenance=provenance,
                    )
                    if sed_row is not None:
                        candidate_rows.append(sed_row)
                except Exception as exc:
                    query_failed = True
                    if progress_callback and idx == 1:
                        progress_callback(f"[SED] {table_name} first lookup failed: {exc}")
                finally:
                    _sleep_after_sed_request()
        except Exception as exc:
            query_failed = True
            if progress_callback and idx == 1:
                progress_callback(f"[SED] HSA unavailable: {exc}")

        rows.extend(candidate_rows)
        if candidate_rows:
            statuses[cid] = "partial" if query_failed else "catalog_detection"
        else:
            statuses[cid] = "query_error" if query_failed else "catalog_no_match"
    return _fetch_result(rows, statuses)


def query_atlasgal_laboca_photometry(
    df: pd.DataFrame,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Fetch uniform ATLASGAL 870 micron catalog photometry from ESO TAP."""

    rows: list[dict] = []
    statuses = _all_candidate_status(df, "query_error")
    try:
        import pyvo

        service = pyvo.dal.TAPService("https://archive.eso.org/tap_cat")
    except Exception:
        return _fetch_result(rows, statuses)
    total = len(df)
    for idx, (_, item) in enumerate(df.iterrows(), start=1):
        cid = _candidate_id_for_row(item)
        if progress_callback and (idx == 1 or idx % 100 == 0 or idx == total):
            progress_callback(f"[SED] apex_laboca {idx}/{total}")
        target_ra, target_dec, coordinate_method = _archive_query_position(
            item,
            epoch_jyear=2009.0,
        )
        if target_ra is None or target_dec is None:
            statuses[cid] = "query_error"
            continue
        try:
            radius_arcsec = 19.2
            radius_deg = radius_arcsec / 3600.0
            cos_dec = max(abs(math.cos(math.radians(target_dec))), 1.0e-3)
            ra_half_width = radius_deg / cos_dec
            ra_min = (target_ra - ra_half_width) % 360.0
            ra_max = (target_ra + ra_half_width) % 360.0
            if ra_min <= ra_max:
                ra_clause = f"RA BETWEEN {ra_min:.10f} AND {ra_max:.10f}"
            else:
                ra_clause = (
                    f"(RA >= {ra_min:.10f} OR RA <= {ra_max:.10f})"
                )
            query = (
                "SELECT TOP 5 ATLAS_NAME, GC_ID, RA, DE, FLUX, "
                "INT_FLUX, SNR, MAJOR_FWHM, MINOR_FWHM, PA "
                "FROM ATLASGAL_V1 "
                f"WHERE {ra_clause} AND "
                f"DE BETWEEN {target_dec - radius_deg:.10f} "
                f"AND {target_dec + radius_deg:.10f}"
            )
            result_table = service.search(query).to_table()
            result = (
                result_table.to_pandas()
                if hasattr(result_table, "to_pandas")
                else pd.DataFrame(result_table)
            )
            match, separation, separations = _nearest_archive_row(
                result,
                target_ra_deg=target_ra,
                target_dec_deg=target_dec,
                radius_arcsec=radius_arcsec,
            )
            if match is None:
                statuses[cid] = "catalog_no_match"
                continue
            flux_jy = _row_float_value(match, ("FLUX", "flux"))
            if flux_jy is None or flux_jy <= 0:
                statuses[cid] = "catalog_no_match"
                continue
            snr = _row_float_value(match, ("SNR", "snr"))
            flux_error_jy = flux_jy / snr if snr is not None and snr > 0 else None
            source_id = _clean_text(_row_value(match, ("ATLAS_NAME", "GC_ID")))
            quality = [
                "confusion_risk",
                "eso_tap_direct",
                "atlasgal_peak_flux_jy_per_beam",
                "diagnostic_only",
            ]
            if len(separations) > 1:
                quality.append("multiple_catalog_matches")
            provenance = {
                "catalog": "ATLASGAL_V1",
                "catalog_object_id": source_id or None,
                "selected_sep_arcsec": separation,
                "match_count_returned": len(separations),
                "coordinate_method": coordinate_method,
                "snr": snr,
                "integrated_flux_jy": _row_float_value(match, ("INT_FLUX", "int_flux")),
                "major_fwhm_arcsec": _row_float_value(match, ("MAJOR_FWHM", "major_fwhm")),
                "minor_fwhm_arcsec": _row_float_value(match, ("MINOR_FWHM", "minor_fwhm")),
            }
            sed_row = _direct_flux_sed_row(
                candidate_id=cid,
                payload=item.to_dict(),
                source="APEX",
                band="LABOCA870",
                flux_jy=flux_jy,
                flux_error_jy=flux_error_jy,
                separation_arcsec=separation,
                catalog_release="ATLASGAL_V1",
                source_object_id=source_id or None,
                observation_id=None,
                instrument="LABOCA",
                quality_flags=quality,
                provenance=provenance,
            )
            if sed_row is not None:
                rows.append(sed_row)
                statuses[cid] = "catalog_detection"
            else:
                statuses[cid] = "catalog_no_match"
        except Exception as exc:
            statuses[cid] = "query_error"
            if progress_callback and idx == 1:
                progress_callback(f"[SED] ATLASGAL first lookup failed: {exc}")
        finally:
            _sleep_after_sed_request()
    return _fetch_result(rows, statuses)


CATALOG_FETCHERS = {
    "payload": rows_from_candidate_frame,
    "allwise": query_irsa_allwise_photometry,
    "gaia_gspc": query_gaia_gspc_photometry,
    "gaia_xp": query_gaia_xp_sampled,
    "galex": lambda df, progress_callback=None: query_vizier_source(df, "galex", progress_callback=progress_callback),
    "catwise": lambda df, progress_callback=None: query_vizier_source(df, "catwise", progress_callback=progress_callback),
    "nsc": query_nsc_photometry,
    "ps1": query_ps1_mean_photometry,
    "sdss": lambda df, progress_callback=None: query_vizier_source(df, "sdss", progress_callback=progress_callback),
    "skymapper": lambda df, progress_callback=None: query_vizier_source(df, "skymapper", progress_callback=progress_callback),
    "des": query_des_photometry,
    "decaps": query_decaps_photometry,
    "ukidss": lambda df, progress_callback=None: query_vizier_source(df, "ukidss", progress_callback=progress_callback),
    "vista": lambda df, progress_callback=None: query_vizier_source(df, "vista", progress_callback=progress_callback),
    "vhs": lambda df, progress_callback=None: query_vizier_source(df, "vhs", progress_callback=progress_callback),
    "viking": lambda df, progress_callback=None: query_vizier_source(df, "viking", progress_callback=progress_callback),
    "vphas": lambda df, progress_callback=None: query_vizier_source(df, "vphas", progress_callback=progress_callback),
    "swift_uvot": lambda df, progress_callback=None: query_vizier_source(df, "swift_uvot", progress_callback=progress_callback),
    "xmm_om": lambda df, progress_callback=None: query_vizier_source(df, "xmm_om", progress_callback=progress_callback),
    "spitzer": query_irsa_spitzer_photometry,
    "akari": lambda df, progress_callback=None: query_flux_catalog_source(df, "akari", progress_callback=progress_callback),
    "iras": lambda df, progress_callback=None: query_flux_catalog_source(df, "iras", progress_callback=progress_callback),
    "herschel": query_herschel_irsa_hsa_photometry,
    "apex_laboca": query_atlasgal_laboca_photometry,
}

SED_SOURCE_FETCH_SIGNATURES.update(
    {
        "allwise": SedFetchSignature(
            catalog_release="irsa:allwise_p3as_psd",
            adapter_version="irsa-allwise-v1",
            match_policy_version="pm-propagated-nearest-3arcsec-v1",
            coordinate_epoch="gaia-ref-to-j2010.5",
            quality_policy_version="allwise-profilefit-quality-v1",
        ),
        "spitzer": SedFetchSignature(
            catalog_release="irsa:slphotdr4",
            adapter_version="irsa-seip-source-list-v1",
            match_policy_version="pm-propagated-nearest-3arcsec-v1",
            coordinate_epoch="gaia-ref-to-j2008.0",
            quality_policy_version="seip-robust-flux-v1",
        ),
        "herschel": SedFetchSignature(
            catalog_release="irsa:hppsc2+hsa:spire-point-source",
            adapter_version="irsa-hppsc2-hsa-spire-v1",
            match_policy_version="pm-propagated-band-radius-v1",
            coordinate_epoch="gaia-ref-to-j2011.0",
            quality_policy_version="hppsc2-spire-quality-v1",
        ),
        "apex_laboca": SedFetchSignature(
            catalog_release="eso:ATLASGAL_V1",
            adapter_version="eso-atlasgal-v1",
            match_policy_version="pm-propagated-nearest-19p2arcsec-v1",
            coordinate_epoch="gaia-ref-to-j2009.0",
            quality_policy_version="atlasgal-diagnostic-v1",
        ),
    }
)

ALL_CATALOG_SOURCES = tuple(CATALOG_FETCHERS)
FAR_IR_CATALOG_SOURCES = ("akari", "iras", "herschel", "apex_laboca")
BROAD_CLASSIFICATION_SED_SOURCES = ("payload", "allwise", "ps1", "skymapper", "sdss")
# Full-catalog acquisition is explicit because it is a long-running external
# operation.  Routine review/pipeline work retains the bounded broad profile;
# callers that want the complete registry must pass ``all``.
DEFAULT_PIPELINE_SED_SOURCES = BROAD_CLASSIFICATION_SED_SOURCES


def resolve_sed_sources(sources: Iterable[str] | str = "default") -> tuple[str, ...]:
    if isinstance(sources, str):
        text = sources.strip().lower()
        if text in {"", "default", "pipeline"}:
            return DEFAULT_PIPELINE_SED_SOURCES
        if text == "all":
            return ALL_CATALOG_SOURCES
        if text in {"broad", "classification", "broad-classification"}:
            return BROAD_CLASSIFICATION_SED_SOURCES
        if text in {"far_ir", "far-ir", "farir"}:
            return FAR_IR_CATALOG_SOURCES
        requested = tuple(x.strip().lower() for x in text.split(",") if x.strip())
    else:
        requested = tuple(str(x).strip().lower() for x in sources if str(x).strip())
    return requested


def _normalize_source_name(value: object) -> str:
    text = str(value or "").strip().lower().replace("_", " ")
    return " ".join(text.split())


def _sed_source_key_for_row_source(
    value: object,
    catalog_release: object | None = None,
) -> str | None:
    norm = _normalize_source_name(value)
    if not norm:
        return None
    release = _normalize_source_name(catalog_release)
    if norm == "allwise" and "allwise p3as psd" in release:
        return "allwise"
    if norm in _PAYLOAD_SED_SOURCES:
        return "payload"
    return _SED_ROW_SOURCE_TO_KEY.get(norm)


def _sed_source_label(source_key: str) -> str:
    return SED_SOURCE_LABELS.get(str(source_key).strip().lower(), str(source_key))


def _sed_source_row_summary(source_key: str, rows: pd.DataFrame) -> dict[str, object]:
    if rows is None or rows.empty:
        return {
            "key": source_key,
            "label": _sed_source_label(source_key),
            "status": "not_queried",
            "n_rows": 0,
            "source_names": [],
            "bands": [],
            "storage": "",
            "message": "",
        }
    source_names = sorted(str(x) for x in rows.get("source", pd.Series(dtype=object)).dropna().unique())
    bands = sorted(str(x) for x in rows.get("band", pd.Series(dtype=object)).dropna().unique())
    return {
        "key": source_key,
        "label": _sed_source_label(source_key),
        "status": "hit",
        "n_rows": int(len(rows)),
        "source_names": source_names,
        "bands": bands,
        "storage": "",
        "message": "",
    }


def _sed_source_rows_for_key(rows: pd.DataFrame | None, candidate_id: str, source_key: str) -> pd.DataFrame:
    if rows is None or rows.empty or "source" not in rows.columns:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).copy()
    if "candidate_id" in frame.columns:
        frame = frame[frame["candidate_id"].astype(str) == str(candidate_id)].copy()
    if frame.empty:
        return pd.DataFrame()
    releases = frame.get("catalog_release", pd.Series(None, index=frame.index))
    key_mask = pd.Series(
        [
            _sed_source_key_for_row_source(source, release) == source_key
            for source, release in zip(frame["source"], releases)
        ],
        index=frame.index,
    )
    return frame.loc[key_mask].copy()


def sed_source_statuses(
    candidate_id: str,
    *,
    payload: dict | pd.Series | None = None,
    external_rows: pd.DataFrame | Iterable[dict] | None = None,
    sources: Iterable[str] | str = "default",
) -> list[dict[str, object]]:
    """Return per-source SED fetch provenance for one candidate.

    Status values are:
    - ``hit``: persisted SED rows exist, or a cache hit has rows.
    - ``partial``: some rows exist, but one or more source requests failed.
    - ``catalog_no_match``: the catalog was queried but observation coverage is
      not yet established.
    - ``covered_no_detection``: valid archive coverage exists without a catalog
      or image detection.
    - ``outside_footprint``: the source has deterministic sky coverage that
      excludes this candidate.
    - ``error``: the query failed and remains retryable.
    - ``not_queried``: neither persisted rows nor a per-source cache entry exists.
    - ``unknown``: a cache entry exists but cannot be interpreted.
    """
    cid = str(candidate_id)
    requested = resolve_sed_sources(sources)
    persisted = pd.DataFrame(external_rows) if external_rows is not None else pd.DataFrame()

    payload_rows = pd.DataFrame()
    if payload is not None and "payload" in requested:
        payload_dict = dict(payload) if not isinstance(payload, dict) else payload
        payload_rows = rows_from_payload(payload_dict, candidate_id=cid, extinction_mode="observed")
        if not payload_rows.empty:
            payload_rows = payload_rows[
                payload_rows["source"].astype(str) != "AllWISE"
            ].copy()
        persisted_payload_rows = _sed_source_rows_for_key(persisted, cid, "payload")
        if not persisted_payload_rows.empty:
            payload_rows = (
                pd.concat([payload_rows, persisted_payload_rows], ignore_index=True)
                if not payload_rows.empty
                else persisted_payload_rows
            )
            payload_rows = payload_rows.drop_duplicates(
                subset=[c for c in ["candidate_id", "source", "band", "quality_flags"] if c in payload_rows.columns],
                keep="first",
            )

    statuses: list[dict[str, object]] = []
    for source_key in requested:
        key = str(source_key).strip().lower()
        if not key:
            continue
        if key == "payload":
            summary = _sed_source_row_summary(key, payload_rows)
            if summary["status"] == "hit":
                summary["storage"] = "payload"
            statuses.append(summary)
            continue

        db_rows = _sed_source_rows_for_key(persisted, cid, key)
        if not db_rows.empty:
            summary = _sed_source_row_summary(key, db_rows)
            summary["storage"] = "review_db"
            statuses.append(summary)
            continue

        cache, cache_error = _read_sed_source_cache_with_error(key)
        if cache_error:
            statuses.append({
                "key": key,
                "label": _sed_source_label(key),
                "status": "unknown",
                "n_rows": 0,
                "source_names": [],
                "bands": [],
                "storage": "cache",
                "message": cache_error,
            })
            continue
        if cache.empty:
            statuses.append(_sed_source_row_summary(key, pd.DataFrame()))
            continue

        cache_rows = cache[cache["_cache_candidate_id"].astype(str) == cid].copy()
        if not cache_rows.empty and payload is not None:
            payload_series = pd.Series(dict(payload))
            valid_signature = _cache_signature_mask(
                cache_rows,
                key,
                {cid: _candidate_astrometry_hash(payload_series)},
            )
            cache_rows = cache_rows.loc[valid_signature].copy()
        if cache_rows.empty:
            statuses.append(_sed_source_row_summary(key, pd.DataFrame()))
            continue

        cache_status = (
            cache_rows["_cache_status"].fillna("hit").astype(str).str.strip().str.lower()
            if "_cache_status" in cache_rows.columns
            else pd.Series("hit", index=cache_rows.index)
        )
        measurement_rows = cache_rows.loc[
            cache_status.isin(SED_CACHE_TERMINAL_STATUSES | {"partial"})
            & cache_rows.get("band", pd.Series(None, index=cache_rows.index)).notna()
        ].copy()
        if not measurement_rows.empty:
            summary = _sed_source_row_summary(key, measurement_rows.reindex(columns=SED_COLUMNS))
            if (cache_status == "partial").any():
                summary["status"] = "partial"
                summary["message"] = "some measurements cached; source fetch remains retryable"
            summary["storage"] = "cache"
            statuses.append(summary)
        elif cache_status.isin(SED_CACHE_TERMINAL_STATUSES).any():
            terminal_status = next(
                (
                    status
                    for status in cache_status.astype(str)
                    if status in SED_CACHE_TERMINAL_STATUSES
                ),
                "miss",
            )
            terminal_messages = {
                "miss": "queried; no catalog match",
                "catalog_no_match": "queried; no catalog match; coverage not established",
                "outside_footprint": "outside the catalog footprint",
                "not_observed": "archive search found no observation coverage",
                "covered_no_detection": "valid coverage; no catalog or image detection",
                "upper_limit": "valid coverage; upper limit measured",
                "ambiguous_counterpart": "catalog counterpart is ambiguous",
                "unusable_measurement": "measurement exists but failed quality validation",
                "reduction_required": "covered product requires instrument-specific reduction",
            }
            statuses.append({
                "key": key,
                "label": _sed_source_label(key),
                "status": terminal_status,
                "n_rows": 0,
                "source_names": [],
                "bands": [],
                "storage": "cache",
                "message": terminal_messages.get(
                    terminal_status,
                    terminal_status.replace("_", " "),
                ),
            })
        elif cache_status.isin(SED_CACHE_RETRYABLE_STATUSES).any():
            retry_status = next(
                (
                    status
                    for status in cache_status.astype(str)
                    if status in SED_CACHE_RETRYABLE_STATUSES
                ),
                "error",
            )
            statuses.append({
                "key": key,
                "label": _sed_source_label(key),
                "status": retry_status,
                "n_rows": 0,
                "source_names": [],
                "bands": [],
                "storage": "cache",
                "message": "archive query failed; retryable",
            })
        else:
            statuses.append({
                "key": key,
                "label": _sed_source_label(key),
                "status": "unknown",
                "n_rows": 0,
                "source_names": [],
                "bands": [],
                "storage": "cache",
                "message": "cache entry has no hit/miss status",
            })
    return statuses


SED_FETCH_MANIFEST_COLUMNS = [
    "candidate_id",
    "source_key",
    "source_label",
    "status",
    "n_rows",
    "cache_updated_at",
    "is_complete",
    "fetch_policy_version",
]


def build_sed_fetch_manifest(
    df: pd.DataFrame,
    *,
    sources: Iterable[str] | str = "default",
    fetched_rows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build an auditable candidate-by-source completion matrix."""
    candidate_ids = list(dict.fromkeys(_sed_candidate_ids(pd.DataFrame(df)).astype(str)))
    requested = resolve_sed_sources(sources)
    fetched = pd.DataFrame(fetched_rows) if fetched_rows is not None else pd.DataFrame()
    fetched_counts: dict[tuple[str, str], int] = {}
    if not fetched.empty and {"candidate_id", "source"}.issubset(fetched.columns):
        keyed = fetched[["candidate_id", "source"]].copy()
        keyed["candidate_id"] = keyed["candidate_id"].astype(str)
        releases = fetched.get("catalog_release", pd.Series(None, index=fetched.index))
        keyed["source_key"] = [
            _sed_source_key_for_row_source(source, release)
            for source, release in zip(keyed["source"], releases)
        ]
        fetched_counts = {
            (str(cid), str(source_key)): int(count)
            for (cid, source_key), count in keyed.dropna(subset=["source_key"]).groupby(
                ["candidate_id", "source_key"],
                dropna=False,
            ).size().items()
        }

    records: list[dict[str, object]] = []
    for source_key in requested:
        key = str(source_key).strip().lower()
        cache = pd.DataFrame()
        if key != "payload":
            cache, _cache_error = _read_sed_source_cache_with_error(key)
            if not cache.empty:
                cache = cache[cache["_cache_candidate_id"].astype(str).isin(candidate_ids)].copy()
        grouped_cache = {
            str(cid): group
            for cid, group in cache.groupby("_cache_candidate_id", sort=False)
        } if not cache.empty else {}

        for cid in candidate_ids:
            n_rows = int(fetched_counts.get((cid, key), 0))
            updated_at = ""
            if key == "payload":
                status = "hit" if n_rows else "miss"
            else:
                candidate_cache = grouped_cache.get(cid)
                if candidate_cache is None or candidate_cache.empty:
                    status = "hit" if n_rows else "not_queried"
                else:
                    statuses = set(
                        candidate_cache.get(
                            "_cache_status",
                            pd.Series("hit", index=candidate_cache.index),
                        ).fillna("hit").astype(str).str.strip().str.lower()
                    )
                    cache_measurements = candidate_cache.get(
                        "band", pd.Series(None, index=candidate_cache.index)
                    ).notna()
                    n_rows = max(n_rows, int(cache_measurements.sum()))
                    unrecognized = statuses - (
                        SED_CACHE_TERMINAL_STATUSES | SED_CACHE_RETRYABLE_STATUSES
                    )
                    retryable = statuses & SED_CACHE_RETRYABLE_STATUSES
                    terminal = statuses & SED_CACHE_TERMINAL_STATUSES
                    if unrecognized:
                        status = "unknown"
                    elif "partial" in retryable or (retryable and terminal):
                        status = "partial"
                    elif retryable:
                        status = next(
                            (
                                candidate_status
                                for candidate_status in ("query_error", "error")
                                if candidate_status in retryable
                            ),
                            "partial",
                        )
                    else:
                        status = next(
                            (
                                candidate_status
                                for candidate_status in SED_MANIFEST_TERMINAL_STATUS_PRIORITY
                                if candidate_status in terminal
                            ),
                            "unknown",
                        )
                    if "_cache_updated_at" in candidate_cache.columns:
                        times = candidate_cache["_cache_updated_at"].dropna().astype(str)
                        updated_at = max(times, default="")
            records.append(
                {
                    "candidate_id": cid,
                    "source_key": key,
                    "source_label": _sed_source_label(key),
                    "status": status,
                    "n_rows": n_rows,
                    "cache_updated_at": updated_at,
                    "is_complete": status in SED_CACHE_TERMINAL_STATUSES,
                    "fetch_policy_version": SED_FETCH_POLICY_VERSION,
                }
            )
    return pd.DataFrame(records, columns=SED_FETCH_MANIFEST_COLUMNS)


def validate_sed_fetch_manifest(
    manifest: pd.DataFrame | None,
    df: pd.DataFrame,
    *,
    sources: Iterable[str] | str = "default",
) -> tuple[bool, list[str]]:
    """Verify that *manifest* is the exact completed candidate-source matrix."""
    frame = pd.DataFrame(manifest) if manifest is not None else pd.DataFrame()
    expected_candidates = set(_sed_candidate_ids(pd.DataFrame(df)).astype(str))
    expected_sources = set(resolve_sed_sources(sources))
    required = {
        "candidate_id",
        "source_key",
        "status",
        "is_complete",
        "fetch_policy_version",
    }
    errors: list[str] = []
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        return False, ["missing columns: " + ", ".join(missing_columns)]

    candidate_ids = frame["candidate_id"].astype(str)
    source_keys = frame["source_key"].astype(str).str.strip().str.lower()
    manifest_candidates = set(candidate_ids)
    manifest_sources = set(source_keys)
    if manifest_candidates != expected_candidates:
        errors.append(
            "candidate set mismatch "
            f"(expected={len(expected_candidates)}, manifest={len(manifest_candidates)})"
        )
    if manifest_sources != expected_sources:
        errors.append(
            "source set mismatch "
            f"(expected={sorted(expected_sources)}, manifest={sorted(manifest_sources)})"
        )

    pair_frame = pd.DataFrame(
        {"candidate_id": candidate_ids, "source_key": source_keys},
        index=frame.index,
    )
    duplicate_pairs = int(pair_frame.duplicated().sum())
    if duplicate_pairs:
        errors.append(f"duplicate candidate-source rows: {duplicate_pairs}")
    expected_rows = len(expected_candidates) * len(expected_sources)
    if len(frame) != expected_rows:
        errors.append(f"matrix row count mismatch (expected={expected_rows}, manifest={len(frame)})")

    if expected_sources and not frame.empty:
        source_counts = pair_frame.groupby("source_key").size().to_dict()
        bad_source_counts = {
            key: int(source_counts.get(key, 0))
            for key in sorted(expected_sources)
            if int(source_counts.get(key, 0)) != len(expected_candidates)
        }
        if bad_source_counts:
            errors.append(f"incomplete source columns: {bad_source_counts}")
    if expected_candidates and not frame.empty:
        candidate_counts = pair_frame.groupby("candidate_id").size()
        bad_candidate_count = int((candidate_counts != len(expected_sources)).sum())
        if bad_candidate_count:
            errors.append(f"candidates missing source rows: {bad_candidate_count}")

    statuses = frame["status"].fillna("").astype(str).str.strip().str.lower()
    incomplete_statuses = int((~statuses.isin(SED_CACHE_TERMINAL_STATUSES)).sum())
    if incomplete_statuses:
        errors.append(f"non-terminal statuses: {incomplete_statuses}")
    complete_flags = frame["is_complete"].map(
        lambda value: str(value).strip().lower() in {"1", "true", "t", "yes"}
    )
    incomplete_flags = int((~complete_flags).sum())
    if incomplete_flags:
        errors.append(f"rows not marked complete: {incomplete_flags}")
    wrong_policy = int(
        (frame["fetch_policy_version"].astype(str) != SED_FETCH_POLICY_VERSION).sum()
    )
    if wrong_policy:
        errors.append(f"fetch policy mismatch: {wrong_policy}")
    return not errors, errors


def fetch_sed_photometry(
    df: pd.DataFrame,
    sources: Iterable[str] | str = "default",
    progress_callback: ProgressCallback | None = None,
    *,
    fetch_chunk_size: int = SED_FETCH_CHUNK_SIZE,
    max_attempts: int = SED_FETCH_MAX_ATTEMPTS,
    retry_base_seconds: float = SED_FETCH_RETRY_BASE_SECONDS,
) -> pd.DataFrame:
    requested = resolve_sed_sources(sources)
    frames = []
    for idx, key in enumerate(requested, start=1):
        fetcher = CATALOG_FETCHERS.get(key)
        if fetcher is None:
            if progress_callback:
                progress_callback(f"[SED] skipping unknown source {key}")
            continue
        started = time.perf_counter()
        if progress_callback:
            progress_callback(f"[SED] source {idx}/{len(requested)} {key}: start")
        try:
            part = _fetch_sed_source_with_cache(
                key,
                fetcher,
                df,
                progress_callback=progress_callback,
                chunk_size=fetch_chunk_size,
                max_attempts=max_attempts,
                retry_base_seconds=retry_base_seconds,
            )
        except Exception as exc:
            if progress_callback:
                progress_callback(f"[SED] source {key}: failed ({exc})")
            part = pd.DataFrame(columns=CANONICAL_SED_COLUMNS)
        if part is not None and not part.empty:
            frames.append(part)
        if progress_callback:
            n_rows = 0 if part is None else len(part)
            progress_callback(f"[SED] source {idx}/{len(requested)} {key}: {n_rows} rows in {time.perf_counter() - started:.1f}s")
    if not frames:
        out = pd.DataFrame(columns=CANONICAL_SED_COLUMNS)
    else:
        out = pd.DataFrame(
            [record for frame in frames for record in frame.to_dict("records")],
            columns=CANONICAL_SED_COLUMNS,
        )
        out["_measurement_key"] = _sed_measurement_dedupe_key(out)
        out = (
            out.drop_duplicates(subset=["_measurement_key"], keep="first")
            .drop(columns=["_measurement_key"])
            .reset_index(drop=True)
        )
    if "normalization_version" in out.columns:
        versions = out["normalization_version"].fillna("").astype(str).str.strip()
        upgrade = versions.isin(
            {"", LEGACY_CANONICAL_SED_NORMALIZATION_VERSION}
        )
        out.loc[upgrade, "normalization_version"] = (
            CANONICAL_SED_NORMALIZATION_VERSION
        )
    out = _normalize_sed_json_text_columns(out)
    out.attrs[SED_FETCH_MANIFEST_ATTR] = build_sed_fetch_manifest(
        df,
        sources=requested,
        fetched_rows=out,
    )
    return out
