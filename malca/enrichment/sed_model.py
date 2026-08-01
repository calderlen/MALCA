"""Castelli/Kurucz stellar-atmosphere fitting for broadband SED rows."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import hashlib
import json
import math
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from malca.enrichment.photometric_calibration import (
    PhotometricCalibration,
    ab_calibration,
    mission_quoted_fnu_calibration,
    vega_zero_point_calibration,
)
from malca.enrichment.synthetic_photometry import (
    BandpassUnavailableError,
    FilterResponse,
    ResponseLoader,
    apply_extinction,
    bandpass_quoted_flux_nu_jy,
    bandpass_flux_nu_jy,
    build_response_map,
    response_manifest_hash,
    response_pivot_wavelength_angstrom,
)
from malca.review.sed_storage import (
    canonical_json,
    hash_sed_measurement_set,
    make_sed_fit_run_hash,
    make_sed_input_hash,
    make_sed_input_manifest_hash,
    make_sed_normalization_hash,
    make_sed_result_fit_run_id,
    sed_point_normalization_record,
)


SED_MODEL_FIT_TABLE_NAME = "sed_model_fits"
SED_MODEL_CURVE_TABLE_NAME = "sed_model_curves"
SED_MODEL_POINT_TABLE_NAME = "sed_model_points"
SED_MODEL_FIT_VERSION = "ck04-bandpass-v8"

SED_MODEL_FIT_COLUMNS = [
    "candidate_id",
    "model_family",
    "fit_version",
    "photometry_method",
    "extinction_law",
    "teff_k",
    "teff_err_k",
    "logg",
    "z",
    "av_fixed",
    "av_fit",
    "av_err",
    "rv",
    "apparent_scale",
    "scale",
    "luminosity_lsun",
    "radius_rsun",
    "chi2",
    "reduced_chi2",
    "n_fit_points",
    "n_available_points",
    "n_rejected_points",
    "fit_lambda_min",
    "fit_lambda_max",
    "fit_bands_json",
    "priors_json",
    "fit_param_names_json",
    "fit_param_values_json",
    "fit_covariance_json",
    "fit_covariance_status",
    "robust_objective",
    "measurement_set_hash",
    "candidate_context_hash",
    "response_manifest_hash",
    "calibration_manifest_hash",
    "input_policy_manifest_hash",
    "fit_recipe_hash",
    "model_grid_hash",
    "model_grid_provenance_json",
    "fit_run_hash",
    "fit_run_id",
    "boundary_flags",
    "status",
    "warning",
]

SED_MODEL_CURVE_COLUMNS = [
    "candidate_id",
    "model_family",
    "fit_version",
    "fit_run_hash",
    "fit_run_id",
    "wavelength_angstrom",
    "lambda_l_lambda",
    "flux_lambda",
    "lambda_l_lambda_intrinsic",
    "lambda_l_lambda_observed",
    "flux_lambda_intrinsic",
    "flux_lambda_observed",
    "teff_k",
    "av_fit",
    "scale",
]

SED_MODEL_POINT_COLUMNS = [
    "candidate_id",
    "fit_version",
    "fit_run_hash",
    "fit_run_id",
    "measurement_id",
    "normalization_version",
    "source",
    "band",
    "fit_role",
    "used",
    "exclusion_reason",
    "prediction_status",
    "prediction_reason",
    "observed_flux_nu_jy",
    "observed_flux_nu_jy_err",
    "observed_flux_lambda",
    "observed_flux_lambda_err",
    "observed_lambda_l_lambda",
    "observed_lambda_l_lambda_err",
    "model_flux_nu_jy",
    "model_flux_nu_jy_intrinsic",
    "observed_mag",
    "model_mag",
    "mag_system",
    "residual_sigma",
    "lambda_eff_angstrom",
    "lambda_pivot_angstrom",
    "lambda_mean_angstrom",
    "lambda_nominal_angstrom",
    "lambda_reference_angstrom",
    "lambda_isophotal_angstrom",
    "plot_lambda_angstrom",
    "plot_lambda_kind",
    "model_flux_lambda",
    "model_flux_lambda_intrinsic",
    "model_lambda_l_lambda",
    "model_lambda_l_lambda_intrinsic",
    "svo_filter_id",
    "response_hash",
    "calibration_id",
    "calibration_hash",
    "normalization_hash",
    "normalization_method",
    "normalization_provenance_json",
    "passband_fidelity",
    "fit_policy",
    "quality_flags",
    "fit_sigma_log",
    "fit_sigma_log_stat",
    "fit_sigma_log_systematic",
    "input_hash",
    "correlation_group",
]

MODEL_FAMILY = "Castelli/Kurucz 2004"
LSUN_ERG_S = 3.828e33
PC_CM = 3.085677581491367e18
MIN_FIT_POINTS = 5
DEFAULT_SIGMA_LOG = 0.08
MIN_SIGMA_LOG = 0.02
DEFAULT_LOGT_BOUNDS = (3.54406, 4.699)
FIT_LAMBDA_MIN_ANGSTROM = 3000.0
FIT_LAMBDA_MAX_ANGSTROM = 10000.0
EXTINCTION_LAW = "G23"
DEFAULT_RV = 3.1
SYSTEMATIC_FLOOR_MAG = 0.03
OUTLIER_SIGMA = 5.0
# This normalization includes the response-calibrated flux conversion.  Bump
# the immutable product version whenever that conversion changes so existing
# ledger rows remain reproducible rather than being overwritten in place.
NORMALIZATION_VERSION = "sed-measurement-v8-detector-pivot"
FIT_RECIPE_POLICY_VERSION = "sed-fit-recipe-v6"
CORRELATION_POLICY_VERSION = "survey-group-offsets-v1"
OPTIMIZER_POLICY_VERSION = "least-squares-soft-l1-multistart-v1"
STELLAR_TAIL_POLICY = "rayleigh-jeans-flambda-lambda^-4-native-anchor-v1"
STELLAR_TAIL_MIN_SAMPLES = 32
STELLAR_TAIL_SAMPLES_PER_DECADE = 256
SOURCE_SYSTEMATIC_FLOOR_MAG = {
    "gaia dr3": 0.02,
    "apass": 0.04,
    "pan-starrs": 0.02,
    "sdss": 0.03,
    "skymapper": 0.03,
    "des": 0.02,
    "decaps": 0.03,
    "noirlab nsc dr2": 0.04,
    "2mass": 0.03,
    "ukidss": 0.03,
    "vista/vvv": 0.03,
    "vista/vhs": 0.03,
    "vista/viking": 0.03,
    "vphas+": 0.04,
}

_UV_SOURCE_TOKENS = ("galex", "swift", "xmm-om")
_IR_SOURCE_TOKENS = (
    "allwise",
    "catwise",
    "wise",
    "spitzer",
    "akari",
    "iras",
    "herschel",
    "2mass",
    "ukidss",
    "vista",
    "vhs",
    "viking",
)


class PystellibsSetupError(RuntimeError):
    """Raised when mandatory pystellibs/Kurucz data are not installed."""


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


def _to_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    try:
        if value is None or pd.isna(value):
            return False
    except Exception:
        if value is None:
            return False
    return bool(value)


def _bool_series(values: pd.Series) -> pd.Series:
    """Coerce persisted boolean encodings without treating non-empty text as true."""
    return values.map(_to_bool).astype(bool)


_MEASUREMENT_HASH_FIELDS = (
    "measurement_id",
    "candidate_id",
    "source",
    "catalog_release",
    "source_object_id",
    "exposure_id",
    "instrument",
    "band",
    "epoch_mjd",
    "native_value",
    "native_error",
    "native_unit",
    "observable_kind",
    "mag",
    "mag_err",
    "mag_system",
    "flux_nu_jy",
    "flux_nu_jy_err",
    "sep_arcsec",
    "is_synthetic",
    "is_upper_limit",
    "quality_flags",
)

_CANDIDATE_CONTEXT_FIELDS = (
    "candidate_id",
    "distance_gspphot",
    "distance_pc",
    "dist_pc",
    "parallax",
    "parallax_mas",
    "A_v_3d",
    "av_3d",
    "av_fixed",
    "av50",
    "ebv_3d",
    "e_bv_3d",
    "ag_gspphot",
    "a_g_val",
    "A_v_3d_err",
    "av_3d_err",
    "av_err",
    "av_sigma",
    "av16",
    "A_v_16",
    "av84",
    "A_v_84",
    "logg_gspphot",
    "logg50",
    "logg",
    "mh_gspphot",
    "m_h_gspphot",
    "feh",
    "mh",
    "teff_gspphot",
    "teff50",
    "teff",
    "teff_gspphot_err",
    "teff_err",
    "teff_error",
    "teff_sigma",
    "teff16",
    "teff_gspphot_lower",
    "teff_lower",
    "teff84",
    "teff_gspphot_upper",
    "teff_upper",
    # APASS B eligibility is payload/color dependent.
    "apass_b",
    "Bmag",
    "b_mag",
    "cousins_i",
    "cousins_ic",
    "ic_mag",
    "i_c_mag",
    "Icmag",
    "Ic",
)


def _json_value(value: object) -> object:
    """Return a stable JSON value for provenance hashing."""
    if isinstance(value, (dict, list, tuple)):
        return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (bool, int, str)):
        return value
    return str(value)


def sed_measurement_set_hash(sed_rows: pd.DataFrame, candidate_id: str | None = None) -> str:
    """Hash immutable/native SED inputs without depending on row order."""
    frame = pd.DataFrame() if sed_rows is None else sed_rows.copy()
    if candidate_id is not None and not frame.empty and "candidate_id" in frame.columns:
        frame = frame[frame["candidate_id"].astype(str) == str(candidate_id)].copy()
    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        records.append({name: _json_value(row.get(name)) for name in _MEASUREMENT_HASH_FIELDS})
    encoded = [json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records]
    payload = "\n".join(sorted(encoded)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sed_candidate_context_hash(candidate: pd.Series | dict | None) -> str:
    """Hash every candidate/payload field that can alter fitting or outputs."""

    row = candidate if candidate is not None else {}
    payload = {name: _json_value(row.get(name)) for name in _CANDIDATE_CONTEXT_FIELDS}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@lru_cache(maxsize=8)
def _model_grid_file_hash(path_text: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns  # They intentionally participate in the cache key.
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_library_class_identity(library: object | None) -> str:
    """Return a stable identity for the atmosphere-library implementation.

    ``pystellibs.Kurucz`` is a public re-export of the implementation class
    ``pystellibs.kurucz.Kurucz``.  Selection-time provenance has no instance
    and historically used the former while completed fits used the latter,
    making every completed fit immediately appear stale.  Canonicalize only
    those known aliases; custom/test implementations retain their full class
    identity.
    """
    if library is None:
        return "pystellibs.Kurucz"
    identity = f"{library.__class__.__module__}.{library.__class__.__qualname__}"
    if identity in {"pystellibs.Kurucz", "pystellibs.kurucz.Kurucz"}:
        return "pystellibs.Kurucz"
    return identity


def sed_model_grid_provenance(library: object | None = None) -> dict[str, object]:
    """Describe and content-fingerprint the atmosphere grid used by a fit."""

    try:
        package_version = importlib.metadata.version("pystellibs")
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    source = getattr(library, "source", None) if library is not None else None
    if source is None:
        try:
            spec = importlib.util.find_spec("pystellibs")
        except (ImportError, ValueError):
            spec = None
        if spec is not None and spec.origin:
            root = Path(spec.origin).resolve().parent
            for candidate in (
                root / "libs" / "kurucz2004.grid.fits",
                root / "ezunits" / "libs" / "kurucz2004.grid.fits",
            ):
                if candidate.exists():
                    source = candidate
                    break
    source_path = Path(str(source)).expanduser().resolve() if source else None
    file_size = None
    file_mtime_ns = None
    content_hash = None
    if source_path is not None and source_path.is_file():
        stat = source_path.stat()
        file_size = int(stat.st_size)
        file_mtime_ns = int(stat.st_mtime_ns)
        content_hash = _model_grid_file_hash(str(source_path), file_size, file_mtime_ns)
    library_class = _model_library_class_identity(library)
    return {
        "library_class": library_class,
        "package_version": package_version,
        "source_path": str(source_path) if source_path is not None else None,
        "file_size": file_size,
        "file_mtime_ns": file_mtime_ns,
        "content_sha256": content_hash,
    }


def sed_model_grid_hash(library: object | None = None) -> str:
    return hashlib.sha256(
        json.dumps(
            sed_model_grid_provenance(library),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def sed_fit_recipe_hash(library: object | None = None) -> str:
    """Hash all fit-policy choices that can change a scientific result."""
    recipe = {
        "recipe_policy_version": FIT_RECIPE_POLICY_VERSION,
        "fit_version": SED_MODEL_FIT_VERSION,
        "model_family": MODEL_FAMILY,
        "model_grid_hash": sed_model_grid_hash(library),
        "extinction_law": EXTINCTION_LAW,
        "rv": DEFAULT_RV,
        "fit_lambda_angstrom": [FIT_LAMBDA_MIN_ANGSTROM, FIT_LAMBDA_MAX_ANGSTROM],
        "log10_teff_default_bounds": list(DEFAULT_LOGT_BOUNDS),
        "av_upper_bound_policy": "clip(max(10,av_initial+5),10,30)",
        "minimum_fit_points": MIN_FIT_POINTS,
        "outlier_sigma": OUTLIER_SIGMA,
        "outlier_refit_rounds": 3,
        "optimizer": {
            "policy_version": OPTIMIZER_POLICY_VERSION,
            "method": "scipy.optimize.least_squares",
            "loss": "soft_l1",
            "f_scale": 1.5,
            "x_scale": [0.1, 1.0, 5.0],
            "max_nfev": 160,
            "multistart": ["candidate_initial", "6000K_av0", "alternate_temperature"],
        },
        "covariance": "pinv(JT_J)*reduced_chi2",
        "correlation_policy_version": CORRELATION_POLICY_VERSION,
        "default_systematic_floor_mag": SYSTEMATIC_FLOOR_MAG,
        "source_systematic_floor_mag": SOURCE_SYSTEMATIC_FLOOR_MAG,
        "normalization_version": NORMALIZATION_VERSION,
        "post_fit_stellar_tail": {
            "policy": STELLAR_TAIL_POLICY,
            "scope": "diagnostic_predictions_and_model_curve_only",
            "fit_optimization_uses_tail": False,
            "anchor": "last_native_positive_f_lambda_sample",
            "f_lambda_power_law_index": -4.0,
            "minimum_log_samples_including_anchor": STELLAR_TAIL_MIN_SAMPLES,
            "samples_per_decade": STELLAR_TAIL_SAMPLES_PER_DECADE,
        },
    }
    return hashlib.sha256(
        json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _quality_tokens(value: object) -> set[str]:
    if isinstance(value, dict):
        return {
            str(key).strip().lower()
            for key, enabled in value.items()
            if _to_bool(enabled) or (isinstance(enabled, str) and enabled.strip())
        }
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip().lower() for item in value if str(item).strip()}
    text = str(value or "").strip()
    if not text:
        return set()
    try:
        decoded = json.loads(text)
    except Exception:
        decoded = None
    if decoded is not None and decoded != value:
        return _quality_tokens(decoded)
    return {token for token in re.split(r"[^a-z0-9_+.-]+", text.lower()) if token}


def _row_text(row: pd.Series | dict, name: str) -> str:
    """Return a clean scalar text value without treating ``pd.NA`` as truthy."""
    value = row.get(name)
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _canonical_response_registration(
    row: pd.Series | dict,
) -> tuple[tuple[str, str], object | None]:
    """Resolve the physical response identity from the central SED registry.

    Registered source/band metadata is authoritative even when an imported row
    carries a non-empty but stale filter identifier or magnitude system.  The
    instrument is part of the lookup so NSC measurement-level rows use the
    DECam, Mosaic3, or 90Prime response rather than the mixed object-mean proxy.
    Unregistered external rows retain their explicit response identity.
    """
    from malca.review.sed import bandpass_for

    instrument = _row_text(row, "instrument")
    bandpass = bandpass_for(
        _row_text(row, "source"),
        _row_text(row, "band"),
        instrument=instrument or None,
    )
    if bandpass is not None:
        return (
            str(getattr(bandpass, "svo_filter_id", None) or "").strip(),
            str(getattr(bandpass, "mag_system", "") or "").strip(),
        ), bandpass
    return (
        _row_text(row, "svo_filter_id"),
        _row_text(row, "mag_system"),
    ), None


def _canonical_response_key(row: pd.Series | dict) -> tuple[str, str]:
    """Return the one response key used for both loading and point lookup."""
    return _canonical_response_registration(row)[0]


def _canonicalize_response_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply authoritative registry response metadata to an input frame."""
    if frame.empty:
        return frame
    for column in ("svo_filter_id", "mag_system", "_zero_point_jy"):
        if column not in frame.columns:
            frame[column] = pd.Series(np.nan, index=frame.index, dtype=object)
    if "lambda_reference_angstrom" not in frame.columns:
        frame["lambda_reference_angstrom"] = np.nan
    for idx, row in frame.iterrows():
        key, bandpass = _canonical_response_registration(row)
        frame.at[idx, "svo_filter_id"] = key[0]
        frame.at[idx, "mag_system"] = key[1]
        if bandpass is None:
            continue
        frame.at[idx, "_zero_point_jy"] = getattr(bandpass, "fnu_zero_jy", None)
        if key[1].casefold() == "jy":
            # For quoted monochromatic fluxes, only the mission definition in
            # the registry is canonical.  Never substitute an SVO response
            # wavelength or a generic effective/pivot wavelength.
            reference = _safe_float(getattr(bandpass, "lambda_reference_angstrom", None))
            frame.at[idx, "lambda_reference_angstrom"] = (
                reference if reference is not None else np.nan
            )
    return frame


def _fit_wavelength(row: pd.Series | dict) -> float | None:
    """Return response/reference wavelength for policy only, never model sampling."""
    for name in ("lambda_pivot_angstrom", "lambda_reference_angstrom", "lambda_eff_angstrom"):
        value = _safe_float(row.get(name))
        if value is not None and value > 0:
            return value
    return None


def _display_wavelength(row: pd.Series | dict) -> float | None:
    for name in ("plot_lambda_angstrom", "lambda_pivot_angstrom", "lambda_reference_angstrom", "lambda_eff_angstrom"):
        value = _safe_float(row.get(name))
        if value is not None and value > 0:
            return value
    return None


def _measurement_id_for_row(row: pd.Series | dict) -> str:
    existing_value = row.get("measurement_id")
    try:
        existing_missing = existing_value is None or bool(pd.isna(existing_value))
    except (TypeError, ValueError):
        existing_missing = existing_value is None
    existing = "" if existing_missing else str(existing_value).strip()
    if existing.casefold() not in {"", "none", "nan", "<na>"}:
        return existing
    try:
        from malca.review.sed_storage import (
            canonical_sed_measurement_identity,
            make_sed_measurement_id,
        )

        return make_sed_measurement_id(canonical_sed_measurement_identity(row))
    except (ImportError, ValueError, TypeError):
        pass
    identity = {
        name: _json_value(row.get(name))
        for name in (
            "candidate_id",
            "source",
            "catalog_release",
            "source_object_id",
            "exposure_id",
            "instrument",
            "band",
            "epoch_mjd",
            "native_value",
            "native_unit",
            "mag",
            "mag_system",
            "flux_nu_jy",
        )
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _calibration_for_row(
    row: pd.Series | dict,
    response: FilterResponse,
) -> PhotometricCalibration | None:
    system = str(row.get("mag_system") or "").strip().lower()
    observable = str(row.get("observable_kind") or "").strip().lower()
    calibration_id = str(row.get("calibration_id") or "").strip()
    if observable == "quoted_fnu" or system == "jy":
        # Quoted catalog fluxes are defined at a mission reference wavelength.
        # Pivot/effective wavelengths and SVO WavelengthRef are response
        # metadata, not interchangeable definitions of that catalog observable.
        reference = _safe_float(row.get("lambda_reference_angstrom"))
        if reference is None:
            return None
        try:
            return mission_quoted_fnu_calibration(response.filter_id, reference)
        except ValueError:
            return None
    if system == "ab":
        return ab_calibration(calibration_id=calibration_id or f"{response.filter_id}/AB/3631Jy")
    if system == "vega":
        zero_point = _safe_float(getattr(response, "zero_point_jy", None)) or _safe_float(
            row.get("_zero_point_jy")
        )
        if zero_point is None:
            return None
        return vega_zero_point_calibration(
            zero_point,
            calibration_id=calibration_id or f"{response.filter_id}/Vega/response-zero-point",
        )
    return None


def _model_catalog_flux_nu_jy(
    wavelength_angstrom: np.ndarray,
    flux_lambda: np.ndarray,
    response: FilterResponse,
    row: pd.Series | dict,
) -> float:
    calibration = _calibration_for_row(row, response)
    system = str(row.get("mag_system") or "").strip().lower()
    observable = str(row.get("observable_kind") or "").strip().lower()
    is_quoted_fnu = observable == "quoted_fnu" or system == "jy"
    if is_quoted_fnu and calibration is None:
        raise BandpassUnavailableError(
            f"No mission quoted-Fnu calibration is registered for {response.filter_id!r}."
        )
    if calibration is not None and calibration.observable_kind == "quoted_fnu":
        return bandpass_quoted_flux_nu_jy(
            wavelength_angstrom,
            flux_lambda,
            response,
            calibration,
        )
    return bandpass_flux_nu_jy(wavelength_angstrom, flux_lambda, response)


def _candidate_id_for_row(row: pd.Series, index: object) -> str:
    for col in ("candidate_id", "asas_sn_id", "gaia_id", "source_id"):
        if col in row:
            value = row.get(col)
            try:
                if value is None or pd.isna(value):
                    continue
            except Exception:
                if value is None:
                    continue
            text = str(value).strip()
            if text and text.lower() not in {"nan", "none", "<na>"}:
                return text
    return str(index)


def _first_finite(row: pd.Series | dict, names: Iterable[str]) -> float | None:
    for name in names:
        if name in row:
            value = _safe_float(row.get(name))
            if value is not None:
                return value
    return None


def _distance_pc_from_candidate(row: pd.Series | dict) -> float | None:
    distance = _first_finite(row, ("distance_gspphot", "distance_pc", "dist_pc"))
    if distance is not None and distance > 0:
        return distance
    parallax = _first_finite(row, ("parallax", "parallax_mas"))
    if parallax is not None and parallax > 0:
        return 1000.0 / parallax
    return None


def _extinction_av_from_candidate(row: pd.Series | dict, r_v: float = 3.1) -> float:
    av = _first_finite(row, ("A_v_3d", "av_3d", "av_fixed", "av50"))
    if av is not None and av >= 0:
        return av
    ebv = _first_finite(row, ("ebv_3d", "e_bv_3d"))
    if ebv is not None and ebv >= 0:
        return float(r_v) * ebv
    ag = _first_finite(row, ("ag_gspphot", "a_g_val"))
    if ag is not None and ag >= 0:
        return ag / 0.789
    return 0.0


def _logg_from_candidate(row: pd.Series | dict) -> float:
    logg = _first_finite(row, ("logg_gspphot", "logg50", "logg"))
    if logg is None:
        return 4.5
    return float(np.clip(logg, 0.0, 5.5))


def _z_from_candidate(row: pd.Series | dict) -> float:
    mh = _first_finite(row, ("mh_gspphot", "m_h_gspphot", "feh", "mh"))
    if mh is None:
        return 0.02
    return float(np.clip(0.02 * (10.0 ** mh), 1.0e-4, 0.08))


def _teff_initial_from_candidate(row: pd.Series | dict, bounds: tuple[float, float]) -> float:
    teff = _first_finite(row, ("teff_gspphot", "teff50", "teff"))
    lo, hi = (10.0 ** bounds[0], 10.0 ** bounds[1])
    if teff is None or teff <= 0:
        return float(np.clip(5772.0, lo, hi))
    return float(np.clip(teff, lo, hi))


def _quantity_to_array(value: object) -> np.ndarray:
    if hasattr(value, "to_value"):
        try:
            return np.asarray(value.to_value(), dtype=float)
        except Exception:
            pass
    if hasattr(value, "value"):
        try:
            return np.asarray(value.value, dtype=float)
        except Exception:
            pass
    if hasattr(value, "magnitude"):
        try:
            return np.asarray(value.magnitude, dtype=float)
        except Exception:
            pass
    return np.asarray(value, dtype=float)


def _load_kurucz_library() -> object:
    try:
        pystellibs = importlib.import_module("pystellibs")
    except Exception as exc:
        raise PystellibsSetupError(
            "pystellibs is required for Castelli/Kurucz SED fitting. "
            "Install it and make sure the Kurucz data file kurucz2004.grid.fits is available."
        ) from exc

    _patch_pystellibs_kurucz_libsdir(pystellibs)

    try:
        library = pystellibs.Kurucz()
    except Exception as exc:
        raise PystellibsSetupError(
            "pystellibs.Kurucz could not load kurucz2004.grid.fits. "
            "Install the pystellibs data files so the Kurucz atmosphere grid is available."
        ) from exc

    source = getattr(library, "source", None)
    if source:
        source_path = Path(str(source)).expanduser()
        if str(source).endswith("kurucz2004.grid.fits") and not source_path.exists():
            raise PystellibsSetupError(
                f"pystellibs.Kurucz data file is missing: {source_path}. "
                "Install kurucz2004.grid.fits before running atmosphere fitting."
            )

    if not hasattr(library, "generate_stellar_spectrum"):
        raise PystellibsSetupError("pystellibs.Kurucz does not expose generate_stellar_spectrum().")
    return library


def _patch_pystellibs_kurucz_libsdir(pystellibs: object) -> None:
    """Use the packaged Kurucz grid location when pystellibs points at ezunits/libs.

    The pinned pystellibs package can install ``kurucz2004.grid.fits`` under
    ``pystellibs/libs`` while its config points ``Kurucz`` at
    ``pystellibs/ezunits/libs``.  Patch the module-level libsdir in memory so
    model fitting works without manually editing site-packages.
    """
    package_file = getattr(pystellibs, "__file__", None)
    if not package_file:
        return

    root = Path(str(package_file)).expanduser().parent
    configured_grid = root / "ezunits" / "libs" / "kurucz2004.grid.fits"
    packaged_grid = root / "libs" / "kurucz2004.grid.fits"
    if configured_grid.exists() or not packaged_grid.exists():
        return

    libsdir = str(packaged_grid.parent)
    for module_name in ("pystellibs.config", "pystellibs.kurucz"):
        try:
            module = importlib.import_module(module_name)
            setattr(module, "libsdir", libsdir)
        except Exception:
            continue


def _library_logt_bounds(library: object) -> tuple[float, float]:
    for attr in ("logT", "_logT"):
        if hasattr(library, attr):
            values = _quantity_to_array(getattr(library, attr))
            finite = values[np.isfinite(values)]
            if finite.size:
                return float(np.nanmin(finite)), float(np.nanmax(finite))
    if hasattr(library, "Teff"):
        values = _quantity_to_array(getattr(library, "Teff"))
        finite = values[np.isfinite(values) & (values > 0)]
        if finite.size:
            logt = np.log10(finite)
            return float(np.nanmin(logt)), float(np.nanmax(logt))
    if hasattr(library, "bbox"):
        try:
            bbox = np.asarray(library.bbox(dlogT=0.0, dlogg=0.0), dtype=float)
            vals = bbox[:, 0]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                return float(np.nanmin(vals)), float(np.nanmax(vals))
        except Exception:
            pass
    return DEFAULT_LOGT_BOUNDS


def _clip_to_library_axis(library: object, value: float, *attribute_names: str) -> float:
    for attr in attribute_names:
        if not hasattr(library, attr):
            continue
        values = _quantity_to_array(getattr(library, attr))
        finite = values[np.isfinite(values)]
        if finite.size:
            return float(np.clip(float(value), float(np.nanmin(finite)), float(np.nanmax(finite))))
    return float(value)


def _generate_spectrum(library: object, logt: float, logg: float, z: float) -> tuple[np.ndarray, np.ndarray]:
    try:
        result = library.generate_stellar_spectrum(logt, logg, 0.0, z, raise_extrapolation=False)
    except TypeError:
        result = library.generate_stellar_spectrum(logt, logg, 0.0, z)

    if isinstance(result, tuple) and len(result) == 2:
        wavelength = _quantity_to_array(result[0])
        spectrum = _quantity_to_array(result[1])
    else:
        wavelength = None
        for attr in ("_wavelength", "wavelength", "wave"):
            if hasattr(library, attr):
                wavelength = _quantity_to_array(getattr(library, attr))
                break
        if wavelength is None:
            raise RuntimeError("Kurucz library did not provide a wavelength grid.")
        spectrum = _quantity_to_array(result)

    if spectrum.ndim > 1:
        spectrum = np.squeeze(spectrum)
    if wavelength.ndim > 1:
        wavelength = np.squeeze(wavelength)
    if len(wavelength) != len(spectrum):
        raise RuntimeError("Kurucz wavelength and spectrum arrays have inconsistent lengths.")

    good = np.isfinite(wavelength) & np.isfinite(spectrum) & (wavelength > 0) & (spectrum > 0)
    if not np.any(good):
        raise RuntimeError("Kurucz generated no finite positive spectrum values.")
    wavelength = np.asarray(wavelength[good], dtype=float)
    spectrum = np.asarray(spectrum[good], dtype=float)
    order = np.argsort(wavelength)
    return wavelength[order], spectrum[order]


def _extend_stellar_spectrum_rayleigh_jeans(
    wavelength_angstrom: np.ndarray,
    flux_lambda: np.ndarray,
    maximum_wavelength_angstrom: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend a best-fit stellar spectrum with a conservative far-IR tail.

    Castelli/Kurucz fitting always uses the unmodified native grid.  This
    post-fit helper exists only so diagnostic broad-band predictions and the
    displayed model curve can cover responses whose red edge lies just beyond
    that grid.  The added tail is the Rayleigh--Jeans asymptote
    ``F_lambda proportional to lambda**-4``, anchored exactly and continuously
    to the final native positive ``F_lambda`` sample.
    """

    wave = np.asarray(wavelength_angstrom, dtype=float)
    flux = np.asarray(flux_lambda, dtype=float)
    if wave.shape != flux.shape or wave.ndim != 1 or wave.size < 2:
        raise ValueError("A one-dimensional stellar spectrum with at least two samples is required.")
    if (
        np.any(~np.isfinite(wave))
        or np.any(~np.isfinite(flux))
        or np.any(wave <= 0)
        or np.any(flux <= 0)
        or np.any(np.diff(wave) <= 0)
    ):
        raise ValueError("The stellar spectrum must be finite, positive, and strictly increasing.")

    target = _safe_float(maximum_wavelength_angstrom)
    native_max = float(wave[-1])
    if target is None or target <= native_max:
        return wave.copy(), flux.copy()

    decades = math.log10(target / native_max)
    sample_count = max(
        STELLAR_TAIL_MIN_SAMPLES,
        int(math.ceil(decades * STELLAR_TAIL_SAMPLES_PER_DECADE)) + 1,
    )
    tail_wave = np.geomspace(native_max, target, sample_count)
    # Preserve exact endpoints even if a platform's geomspace rounding changes.
    tail_wave[0] = native_max
    tail_wave[-1] = target
    tail_flux = float(flux[-1]) * np.power(tail_wave / native_max, -4.0)
    return (
        np.concatenate((wave, tail_wave[1:])),
        np.concatenate((flux, tail_flux[1:])),
    )


def _empty_fit_row(candidate_id: str, *, status: str, warning: str, row: pd.Series | dict | None = None) -> dict:
    row = row if row is not None else {}
    av_initial = _extinction_av_from_candidate(row)
    return {
        "candidate_id": str(candidate_id),
        "model_family": MODEL_FAMILY,
        "fit_version": SED_MODEL_FIT_VERSION,
        "photometry_method": "bandpass_integrated",
        "extinction_law": EXTINCTION_LAW,
        "teff_k": np.nan,
        "teff_err_k": np.nan,
        "logg": _logg_from_candidate(row),
        "z": _z_from_candidate(row),
        "av_fixed": av_initial,
        "av_fit": np.nan,
        "av_err": np.nan,
        "rv": DEFAULT_RV,
        "apparent_scale": np.nan,
        "scale": np.nan,
        "luminosity_lsun": np.nan,
        "radius_rsun": np.nan,
        "chi2": np.nan,
        "reduced_chi2": np.nan,
        "n_fit_points": 0,
        "n_available_points": 0,
        "n_rejected_points": 0,
        "fit_lambda_min": np.nan,
        "fit_lambda_max": np.nan,
        "fit_bands_json": "[]",
        "priors_json": "{}",
        "fit_param_names_json": json.dumps(["log10_teff", "av", "log10_apparent_scale"]),
        "fit_param_values_json": "[]",
        "fit_covariance_json": "[]",
        "fit_covariance_status": "unavailable",
        "robust_objective": np.nan,
        "measurement_set_hash": "",
        "candidate_context_hash": sed_candidate_context_hash(row),
        "response_manifest_hash": "",
        "calibration_manifest_hash": "",
        "input_policy_manifest_hash": "",
        "fit_recipe_hash": sed_fit_recipe_hash(),
        "model_grid_hash": "",
        "model_grid_provenance_json": "{}",
        "fit_run_hash": "",
        "fit_run_id": "",
        "boundary_flags": "",
        "status": status,
        "warning": warning,
    }


def _correlation_group(row: pd.Series) -> str:
    source = str(row.get("source") or "").strip().lower()
    if "gaia xp" in source:
        return "gaia_xp"
    if "gaia gspc" in source:
        return "gaia_gspc"
    if "gaia" in source:
        return "gaia_broadband"
    return re.sub(r"[^a-z0-9]+", "_", source).strip("_")


def _point_role_and_reason(row: pd.Series) -> tuple[str, str]:
    source = str(row.get("source") or "").strip().lower()
    band = str(row.get("band") or "").strip().lower()
    flags = _quality_tokens(row.get("quality_flags"))
    normalized_flags = _quality_tokens(row.get("normalized_quality_status"))
    flags |= normalized_flags
    lam = _fit_wavelength(row)
    fit_policy = str(row.get("fit_policy") or "").strip().lower()
    fidelity = str(row.get("passband_fidelity") or "").strip().lower()
    if _to_bool(row.get("is_upper_limit", False)):
        return "upper_limit", "upper_limit"
    if fit_policy in {
        "exclude",
        "never",
        "diagnostic",
        "diagnostic_only",
        "diagnostic-only",
        "display_only",
        "display-only",
    }:
        return "diagnostic", f"fit_policy:{fit_policy}"
    if fidelity in {"mixed_unknown", "unknown_mixed"}:
        return "photosphere", "mixed_instrument_mean"
    if "noirlab nsc" in source:
        instrument = str(row.get("instrument") or "").strip().lower()
        if not instrument or fidelity != "exact":
            return "photosphere", "mixed_instrument_mean"
    if source == "apass" and band == "b" and (
        _to_bool(row.get("apass_b_red_leak_risk")) or "apass_b_red_leak_likely" in flags
    ):
        return "photosphere", "apass_b_red_leak_risk"
    if "gaia xp" in source or "correlated_spectrum" in flags:
        return "correlated_spectrum", "correlated_spectrum"
    if "gaia gspc" in source:
        return "photosphere", "correlated_gaia_synthetic"
    if "non_simultaneous_pointed" in flags or any(token in source for token in ("swift", "xmm-om")):
        return "pointed_photometry", "non_simultaneous_pointed"
    if "halpha" in band or band in {"ha", "h-alpha", "h_alpha"}:
        return "emission_line", "emission_line"
    if any(token in source for token in _UV_SOURCE_TOKENS) or (lam is not None and lam < FIT_LAMBDA_MIN_ANGSTROM):
        return "uv_diagnostic", "uv_diagnostic"
    if any(token in source for token in _IR_SOURCE_TOKENS) or (lam is not None and lam > FIT_LAMBDA_MAX_ANGSTROM):
        role = "dust" if lam is not None and lam >= 300000.0 else "ir_excess"
        return role, "ir_excess_diagnostic"
    if "confusion_risk" in flags:
        return "photosphere", "confusion_risk"
    sep_arcsec = _safe_float(row.get("sep_arcsec"))
    if sep_arcsec is not None and sep_arcsec > 3.0:
        return "photosphere", "crossmatch_separation"
    mag_err = _safe_float(row.get("mag_err"))
    if mag_err is not None and mag_err > 0.5:
        return "photosphere", "large_photometric_error"
    if flags.intersection({"saturated", "bad_quality", "severe_artifact"}):
        return "photosphere", "catalog_quality"
    if lam is None or not (FIT_LAMBDA_MIN_ANGSTROM <= lam <= FIT_LAMBDA_MAX_ANGSTROM):
        return "unknown", "outside_photosphere_range"
    return "photosphere", ""


def _is_stellar_fit_band(row: pd.Series) -> bool:
    role, reason = _point_role_and_reason(row)
    return role == "photosphere" and not reason


def _fnu_from_flam(flux_lambda: float, wavelength_angstrom: float) -> float:
    return float(flux_lambda) * float(wavelength_angstrom) ** 2 / 2.99792458e18 / 1.0e-23


def _flam_from_fnu(flux_nu_jy: float, wavelength_angstrom: float) -> float:
    return float(flux_nu_jy) * 1.0e-23 * 2.99792458e18 / float(wavelength_angstrom) ** 2


def _prepare_candidate_points(
    candidate_id: str,
    candidate: pd.Series | dict,
    sed_rows: pd.DataFrame,
    responses: dict[tuple[str, str], FilterResponse],
    response_failures: dict[tuple[str, str], str],
) -> pd.DataFrame:
    if sed_rows is None or sed_rows.empty or "candidate_id" not in sed_rows.columns:
        return pd.DataFrame()
    frame = sed_rows[sed_rows["candidate_id"].astype(str) == str(candidate_id)].copy()
    if frame.empty:
        return frame
    for text_column in (
        "measurement_id",
        "normalization_version",
        "catalog_release",
        "source_object_id",
        "instrument",
        "exposure_id",
        "native_unit",
        "calibration_id",
        "calibration_hash",
        "response_hash",
        "normalization_hash",
        "passband_fidelity",
        "observable_kind",
        "quality_flags",
        "plot_lambda_kind",
        "response_kind",
        "fit_policy",
        "native_flux_unit",
        "calibration_source",
    ):
        if text_column not in frame.columns:
            frame[text_column] = pd.Series("", index=frame.index, dtype=object)
        else:
            frame[text_column] = frame[text_column].astype(object)
    if "sed_mode" in frame.columns:
        observed = frame["sed_mode"].fillna("").astype(str).str.strip().str.lower().isin({"", "observed"})
        if observed.any():
            frame = frame.loc[observed].copy()

    _canonicalize_response_fields(frame)

    # This is the single observed-measurement conversion used by both the
    # fitter and the review plot.  Keep catalog-native values on the row while
    # deriving one response-calibrated Fnu/display representation.
    from malca.review.sed import prepare_sed_measurement_row

    for idx, row in frame.copy().iterrows():
        response_key = _canonical_response_key(row)
        is_quoted_fnu = (
            _row_text(row, "observable_kind").casefold() == "quoted_fnu"
            or response_key[1].casefold() == "jy"
        )
        canonical_reference = _safe_float(row.get("lambda_reference_angstrom"))
        prepared = prepare_sed_measurement_row(
            row,
            payload=candidate,
            candidate_id=candidate_id,
            # This helper historically copied SVO WavelengthRef into a missing
            # catalog reference.  Suppress that behavior for quoted-Fnu rows;
            # response metadata is attached separately below.
            response=None if is_quoted_fnu else responses.get(response_key),
        )
        if prepared is None:
            continue
        for column, value in prepared.items():
            frame.at[idx, column] = value
        if is_quoted_fnu:
            frame.at[idx, "lambda_reference_angstrom"] = (
                canonical_reference if canonical_reference is not None else np.nan
            )
        response_kind = str(prepared.get("response_kind") or "").strip().lower()
        if not str(frame.at[idx, "passband_fidelity"] if "passband_fidelity" in frame else "").strip():
            if response_kind == "standardized_system_proxy":
                frame.at[idx, "passband_fidelity"] = "standardized_proxy"
            elif response_kind == "mixed_unknown":
                frame.at[idx, "passband_fidelity"] = "mixed_unknown"
            else:
                frame.at[idx, "passband_fidelity"] = "exact"
        if not str(frame.at[idx, "observable_kind"] if "observable_kind" in frame else "").strip():
            system = str(prepared.get("mag_system") or "").strip().lower()
            frame.at[idx, "observable_kind"] = (
                "quoted_fnu" if system == "jy" else "ab_mag" if system == "ab" else "vega_mag"
            )

    roles = frame.apply(_point_role_and_reason, axis=1)
    frame["fit_role"] = [item[0] for item in roles]
    frame["exclusion_reason"] = [item[1] for item in roles]
    frame["prediction_status"] = "pending"
    frame["prediction_reason"] = ""
    frame["correlation_group"] = frame.apply(_correlation_group, axis=1)
    frame["measurement_id"] = frame.apply(_measurement_id_for_row, axis=1)
    # Pre-fit catalog conversions have their own canonical version.  The
    # response-calibrated values below are a distinct immutable product and
    # must never reuse that version label.
    frame["normalization_version"] = NORMALIZATION_VERSION
    frame["observed_flux_nu_jy"] = pd.to_numeric(
        frame.get("flux_nu_jy", pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    frame["observed_flux_nu_jy_err"] = pd.to_numeric(
        frame.get("flux_nu_jy_err", pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    frame["observed_mag"] = pd.to_numeric(
        frame.get("mag", pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    frame["lambda_eff_angstrom"] = pd.to_numeric(
        frame.get("lambda_eff_angstrom", pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    for wavelength_column in (
        "lambda_pivot_angstrom",
        "lambda_mean_angstrom",
        "lambda_nominal_angstrom",
        "lambda_reference_angstrom",
        "lambda_isophotal_angstrom",
        "plot_lambda_angstrom",
    ):
        frame[wavelength_column] = pd.to_numeric(
            frame.get(wavelength_column, pd.Series(np.nan, index=frame.index)), errors="coerce"
        )
    missing_plot_lambda = ~np.isfinite(frame["plot_lambda_angstrom"])
    frame.loc[missing_plot_lambda, "plot_lambda_angstrom"] = [
        _display_wavelength(row)
        for _, row in frame.loc[missing_plot_lambda].iterrows()
    ]
    if "plot_lambda_kind" not in frame:
        frame["plot_lambda_kind"] = ""
    empty_plot_kind = frame["plot_lambda_kind"].fillna("").astype(str).str.strip().eq("")
    quoted_fnu = frame.get("observable_kind", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower().eq("quoted_fnu")
    frame.loc[empty_plot_kind & quoted_fnu, "plot_lambda_kind"] = "mission_reference"
    frame.loc[empty_plot_kind & ~quoted_fnu, "plot_lambda_kind"] = "response_pivot"
    frame["fit_wavelength_angstrom"] = frame.apply(_fit_wavelength, axis=1)

    missing_fnu = ~np.isfinite(frame["observed_flux_nu_jy"])
    if "flux_lambda" in frame.columns:
        flam = pd.to_numeric(frame["flux_lambda"], errors="coerce")
        convertible = missing_fnu & np.isfinite(flam) & np.isfinite(frame["lambda_eff_angstrom"])
        frame.loc[convertible, "observed_flux_nu_jy"] = [
            _fnu_from_flam(value, wave)
            for value, wave in zip(flam.loc[convertible], frame.loc[convertible, "lambda_eff_angstrom"], strict=False)
        ]
        if "flux_lambda_err" in frame.columns:
            flam_err = pd.to_numeric(frame["flux_lambda_err"], errors="coerce")
            convertible_err = convertible & np.isfinite(flam_err)
            frame.loc[convertible_err, "observed_flux_nu_jy_err"] = [
                _fnu_from_flam(value, wave)
                for value, wave in zip(flam_err.loc[convertible_err], frame.loc[convertible_err, "lambda_eff_angstrom"], strict=False)
            ]

    sigma_log = np.full(len(frame), DEFAULT_SIGMA_LOG, dtype=float)
    fnu = frame["observed_flux_nu_jy"].to_numpy(dtype=float)
    fnu_err = frame["observed_flux_nu_jy_err"].to_numpy(dtype=float)
    good_err = np.isfinite(fnu) & (fnu > 0) & np.isfinite(fnu_err) & (fnu_err > 0)
    sigma_log[good_err] = fnu_err[good_err] / fnu[good_err] / math.log(10.0)
    low_snr = good_err & ((fnu_err / np.where(fnu > 0, fnu, np.nan)) > 0.5)
    low_snr_rows = frame.index[low_snr & (frame["fit_role"].to_numpy() == "photosphere") & (frame["exclusion_reason"].astype(str).to_numpy() == "")]
    frame.loc[low_snr_rows, "exclusion_reason"] = "large_photometric_error"
    mag_err = pd.to_numeric(frame.get("mag_err"), errors="coerce").to_numpy(dtype=float)
    good_mag_err = ~good_err & np.isfinite(mag_err) & (mag_err > 0)
    sigma_log[good_mag_err] = 0.4 * mag_err[good_mag_err]
    source_floors_mag = np.asarray([
        SOURCE_SYSTEMATIC_FLOOR_MAG.get(str(source or "").strip().lower(), SYSTEMATIC_FLOOR_MAG)
        for source in frame.get("source", pd.Series("", index=frame.index))
    ], dtype=float)
    explicit_floors = pd.to_numeric(
        frame.get("systematic_floor_mag", pd.Series(np.nan, index=frame.index)), errors="coerce"
    ).to_numpy(dtype=float)
    source_floors_mag = np.where(
        np.isfinite(explicit_floors) & (explicit_floors >= 0), explicit_floors, source_floors_mag
    )
    for position, (_, row) in enumerate(frame.iterrows()):
        if "apass_b_red_leak_unassessed" in _quality_tokens(row.get("quality_flags")):
            source_floors_mag[position] = max(float(source_floors_mag[position]), 0.10)
    floor_log = 0.4 * source_floors_mag
    frame["fit_sigma_log_stat"] = np.clip(sigma_log, MIN_SIGMA_LOG / 2.0, 0.5)
    frame["fit_sigma_log_systematic"] = floor_log
    sigma_log = np.sqrt(np.square(sigma_log) + np.square(floor_log))
    frame["fit_sigma_log"] = np.clip(sigma_log, np.maximum(floor_log, MIN_SIGMA_LOG), 0.5)

    # Re-resolve after measurement canonicalization.  The same helper builds
    # the response-loading manifest and performs the per-point lookup.
    frame["response_key"] = [
        _canonical_response_key(row) for _, row in frame.iterrows()
    ]
    for idx, key in frame["response_key"].items():
        frame.at[idx, "svo_filter_id"] = key[0]
        frame.at[idx, "mag_system"] = key[1]
    for idx, row in frame.iterrows():
        observed_value = _safe_float(row.get("observed_flux_nu_jy"))
        if observed_value is None or observed_value <= 0:
            frame.at[idx, "exclusion_reason"] = str(row.get("exclusion_reason") or "missing_observed_flux")
            frame.at[idx, "observed_flux_nu_jy"] = np.nan
        key = row["response_key"]
        is_quoted_fnu = (
            str(row.get("observable_kind") or "").strip().lower() == "quoted_fnu"
            or str(row.get("mag_system") or "").strip().lower() == "jy"
        )
        mission_calibration_registered = True
        if not key[0]:
            frame.at[idx, "exclusion_reason"] = str(row.get("exclusion_reason") or "missing_filter_id")
            frame.at[idx, "prediction_status"] = "unavailable"
            frame.at[idx, "prediction_reason"] = "missing_filter_id"
        elif is_quoted_fnu:
            reference = _safe_float(row.get("lambda_reference_angstrom"))
            try:
                if reference is None:
                    raise ValueError("missing mission reference wavelength")
                mission_quoted_fnu_calibration(str(key[0]), reference)
            except ValueError:
                mission_calibration_registered = False
                frame.at[idx, "exclusion_reason"] = str(
                    frame.at[idx, "exclusion_reason"] or "missing_mission_calibration"
                )
                frame.at[idx, "prediction_status"] = "unavailable"
                frame.at[idx, "prediction_reason"] = "missing_mission_calibration"
        if key[0] and mission_calibration_registered and key not in responses:
            failure = response_failures.get(key, "response unavailable")
            frame.at[idx, "exclusion_reason"] = str(row.get("exclusion_reason") or f"missing_bandpass:{failure}")
            frame.at[idx, "prediction_status"] = "unavailable"
            frame.at[idx, "prediction_reason"] = f"missing_bandpass:{failure}"
        response = responses.get(key)
        if response is not None:
            frame.at[idx, "response_hash"] = str(response.response_hash or "")
            # This frame is constructing a new fitter normalization.  Current
            # response state therefore overrides any stale pivot copied from a
            # legacy/catalog row.  Rendering an immutable stored normalization
            # snapshot is handled separately by the review/storage path.
            try:
                frame.at[idx, "lambda_pivot_angstrom"] = response_pivot_wavelength_angstrom(response)
            except BandpassUnavailableError:
                pass
            if not is_quoted_fnu and _safe_float(frame.at[idx, "lambda_reference_angstrom"]) is None:
                frame.at[idx, "lambda_reference_angstrom"] = _safe_float(
                    getattr(response, "wavelength_ref_angstrom", None)
                )
            row_observable = str(row.get("observable_kind") or "").strip().lower()
            row_system = str(row.get("mag_system") or "").strip().lower()
            if row_observable == "quoted_fnu" or row_system == "jy":
                preferred_plot = _safe_float(frame.at[idx, "lambda_reference_angstrom"])
                preferred_kind = "mission_reference"
            else:
                preferred_plot = _safe_float(frame.at[idx, "lambda_pivot_angstrom"])
                preferred_kind = "response_pivot"
            if preferred_plot is not None:
                frame.at[idx, "plot_lambda_angstrom"] = preferred_plot
                frame.at[idx, "plot_lambda_kind"] = preferred_kind
        calibration_id = str(row.get("calibration_id") or "").strip() or (
            f"{key[0]}:{str(row.get('observable_kind') or row.get('mag_system') or 'unknown').strip().lower()}"
        )
        frame.at[idx, "calibration_id"] = calibration_id
        calibration = _calibration_for_row(frame.loc[idx], response) if response is not None else None
        if calibration is not None:
            # In particular, quoted-Jy rows must identify the mission
            # reference-spectrum calibration that is actually hashed, not a
            # generic ``filter:quoted_fnu`` placeholder.
            calibration_id = calibration.calibration_id
            frame.at[idx, "calibration_id"] = calibration_id
        if response is not None and is_quoted_fnu and calibration is None:
            frame.at[idx, "exclusion_reason"] = str(
                frame.at[idx, "exclusion_reason"] or "missing_mission_calibration"
            )
            frame.at[idx, "prediction_status"] = "unavailable"
            frame.at[idx, "prediction_reason"] = "missing_mission_calibration"
        calibration_payload = {
            "calibration_id": calibration_id,
            "observable_kind": str(row.get("observable_kind") or ""),
            "mag_system": str(row.get("mag_system") or ""),
            "zero_point_jy": _safe_float(getattr(response, "zero_point_jy", None)),
            "wavelength_reference_angstrom": _safe_float(frame.at[idx, "lambda_reference_angstrom"]),
            "response_hash": str(getattr(response, "response_hash", "") or ""),
        }
        calibration_hash = (
            calibration.calibration_hash
            if calibration is not None
            else hashlib.sha256(
                json.dumps(calibration_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        )
        frame.at[idx, "calibration_hash"] = calibration_hash
        calibration_provenance = {
            "calibration_id": calibration_id,
            "calibration_version": getattr(calibration, "version", None),
            "calibration_contract": getattr(calibration, "forward_contract", None),
            "reference_spectrum": getattr(calibration, "reference_spectrum", None),
            "reference_temperature_k": getattr(calibration, "reference_temperature_k", None),
            "svo_filter_id": key[0],
            "fit_policy": _row_text(frame.loc[idx], "fit_policy"),
            "passband_fidelity": _row_text(frame.loc[idx], "passband_fidelity"),
        }
        response_hash = _row_text(frame.loc[idx], "response_hash")
        normalization_identity = {
            "normalization_contract": NORMALIZATION_VERSION,
            "response_hash": response_hash,
            "calibration_hash": calibration_hash,
            **calibration_provenance,
        }
        normalization_identity_hash = hashlib.sha256(
            json.dumps(normalization_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        normalization_version = f"{NORMALIZATION_VERSION}:{normalization_identity_hash[:24]}"
        normalization_method = "fitter_bandpass_calibrated_v8"
        frame.at[idx, "normalization_version"] = normalization_version
        frame.at[idx, "normalization_method"] = normalization_method
        frame.at[idx, "normalization_provenance_json"] = canonical_json(calibration_provenance)
        normalization_hash = make_sed_normalization_hash(
            sed_point_normalization_record(frame.loc[idx])
        )
        frame.at[idx, "normalization_hash"] = normalization_hash
    frame["used"] = (frame["fit_role"] == "photosphere") & (frame["exclusion_reason"].astype(str) == "")
    return frame


def _candidate_response_manifest_hash(
    points: pd.DataFrame,
    responses: dict[tuple[str, str], FilterResponse],
) -> str:
    keys = set(points.get("response_key", pd.Series(dtype=object)).dropna().tolist())
    subset = {key: responses[key] for key in keys if key in responses}
    return response_manifest_hash(subset)


def _candidate_calibration_manifest_hash(points: pd.DataFrame) -> str:
    records = sorted(
        {
            (
                str(row.get("calibration_id") or ""),
                str(row.get("calibration_hash") or ""),
                str(row.get("normalization_version") or ""),
                str(row.get("normalization_hash") or ""),
            )
            for _, row in points.iterrows()
        }
    )
    return hashlib.sha256(
        json.dumps(records, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _candidate_input_policy_manifest_hash(points: pd.DataFrame) -> str:
    if points is None or points.empty:
        return make_sed_input_manifest_hash([])
    return make_sed_input_manifest_hash(points.to_dict(orient="records"))


def sed_fit_input_state(
    candidates: pd.DataFrame,
    sed_rows: pd.DataFrame,
    *,
    library: object | None = None,
    response_loader: ResponseLoader | None = None,
) -> pd.DataFrame:
    """Build cache-only fit provenance without generating Kurucz spectra."""

    candidate_frame = pd.DataFrame() if candidates is None else candidates.copy()
    photometry = pd.DataFrame() if sed_rows is None else sed_rows.copy()
    if not photometry.empty:
        _canonicalize_response_fields(photometry)
    filter_pairs = [
        _canonical_response_key(row) for _, row in photometry.iterrows()
    ]
    responses, failures = build_response_map(
        filter_pairs,
        allow_download=False,
        response_loader=response_loader,
    )
    candidate_map: dict[str, pd.Series | dict] = {}
    for idx, row in candidate_frame.iterrows():
        candidate_map[_candidate_id_for_row(row, idx)] = row
    for candidate_id in photometry.get("candidate_id", pd.Series(dtype=str)).dropna().astype(str).unique():
        candidate_map.setdefault(candidate_id, {"candidate_id": candidate_id})
    recipe_hash = sed_fit_recipe_hash(library)
    grid_hash = sed_model_grid_hash(library)
    records: list[dict[str, str]] = []
    for candidate_id, candidate in candidate_map.items():
        points = _prepare_candidate_points(
            candidate_id,
            candidate,
            photometry,
            responses,
            failures,
        )
        records.append(
            {
                "candidate_id": str(candidate_id),
                "measurement_set_hash": hash_sed_measurement_set(points),
                "candidate_context_hash": sed_candidate_context_hash(candidate),
                "response_manifest_hash": _candidate_response_manifest_hash(points, responses),
                "calibration_manifest_hash": _candidate_calibration_manifest_hash(points),
                "input_policy_manifest_hash": _candidate_input_policy_manifest_hash(points),
                "fit_recipe_hash": recipe_hash,
                "model_grid_hash": grid_hash,
            }
        )
    return pd.DataFrame(records)


def _fit_run_hash(
    candidate_id: str,
    measurement_hash: str,
    candidate_context_hash: str,
    response_hash: str,
    calibration_hash: str,
    input_policy_hash: str,
    recipe_hash: str,
    model_grid_hash: str,
) -> str:
    return make_sed_fit_run_hash(
        {
            "candidate_id": str(candidate_id),
            "model_family": MODEL_FAMILY,
            "fit_version": SED_MODEL_FIT_VERSION,
            "photometry_method": "bandpass_integrated",
            "extinction_law": EXTINCTION_LAW,
            "measurement_set_hash": measurement_hash,
            "candidate_context_hash": candidate_context_hash,
            "response_manifest_hash": response_hash,
            "calibration_manifest_hash": calibration_hash,
            "input_policy_manifest_hash": input_policy_hash,
            "fit_recipe_hash": recipe_hash,
            "model_grid_hash": model_grid_hash,
        }
    )


def _format_fit_point_labels(points: pd.DataFrame, *, limit: int = 6) -> str:
    if points is None or points.empty:
        return ""
    grouped: dict[str, list[str]] = {}
    for _, row in points.iterrows():
        source = str(row.get("source") or "").strip()
        band = str(row.get("band") or "").strip()
        source_key = source or "SED"
        if source_key not in grouped:
            grouped[source_key] = []
        if band and band not in grouped[source_key]:
            grouped[source_key].append(band)
    labels = []
    for source, bands in grouped.items():
        if bands:
            labels.append(f"{source} {','.join(bands)}")
        else:
            labels.append(source)
    shown = ", ".join(labels[:limit])
    if len(labels) > limit:
        shown += f", +{len(labels) - limit} more"
    return shown


def _fit_single_candidate(
    candidate_id: str,
    candidate: pd.Series | dict,
    points: pd.DataFrame,
    library: object,
    responses: dict[tuple[str, str], FilterResponse],
    measurement_hash: str,
    response_hash: str,
    calibration_hash: str,
    recipe_hash: str,
    run_hash: str,
    *,
    curve_points: int = 400,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    diagnostics = points.copy()
    if diagnostics.empty:
        usable = pd.DataFrame()
    else:
        used = diagnostics.get("used", pd.Series(False, index=diagnostics.index))
        usable = diagnostics.loc[_bool_series(used)].copy()
    if len(usable) < MIN_FIT_POINTS:
        found = f"found {len(usable)}"
        labels = _format_fit_point_labels(usable)
        if labels:
            found += f": {labels}"
        fit_row = _empty_fit_row(
            candidate_id,
            status="insufficient_data",
            warning=f"Need at least {MIN_FIT_POINTS} bandpass-calibrated photospheric SED points; {found}.",
            row=candidate,
        )
        fit_row["n_available_points"] = int(len(diagnostics))
        fit_row["measurement_set_hash"] = measurement_hash
        fit_row["response_manifest_hash"] = response_hash
        fit_row["calibration_manifest_hash"] = calibration_hash
        fit_row["fit_recipe_hash"] = recipe_hash
        fit_row["fit_run_hash"] = run_hash
        return fit_row, pd.DataFrame(columns=SED_MODEL_CURVE_COLUMNS), _finalize_point_rows(diagnostics, run_hash)

    logg = _clip_to_library_axis(library, _logg_from_candidate(candidate), "logg", "_logg")
    z = _clip_to_library_axis(library, _z_from_candidate(candidate), "Z", "_Z")
    av_initial = _extinction_av_from_candidate(candidate)
    teff_prior = _first_finite(candidate, ("teff_gspphot", "teff50", "teff"))
    teff_prior_sigma_k = _first_finite(
        candidate,
        ("teff_gspphot_err", "teff_err", "teff_error", "teff_sigma"),
    )
    teff16 = _first_finite(candidate, ("teff16", "teff_gspphot_lower", "teff_lower"))
    teff84 = _first_finite(candidate, ("teff84", "teff_gspphot_upper", "teff_upper"))
    if teff_prior_sigma_k is None and teff16 is not None and teff84 is not None and teff84 > teff16:
        teff_prior_sigma_k = 0.5 * (teff84 - teff16)
    has_teff_prior = (
        teff_prior is not None
        and teff_prior > 0
        and teff_prior_sigma_k is not None
        and teff_prior_sigma_k > 0
    )
    av_prior_sigma = _first_finite(candidate, ("A_v_3d_err", "av_3d_err", "av_err", "av_sigma"))
    av16 = _first_finite(candidate, ("av16", "A_v_16"))
    av84 = _first_finite(candidate, ("av84", "A_v_84"))
    if av_prior_sigma is None and av16 is not None and av84 is not None and av84 > av16:
        av_prior_sigma = 0.5 * (av84 - av16)
    has_av_prior = av_prior_sigma is not None and av_prior_sigma > 0
    if has_av_prior:
        av_prior_sigma = max(float(av_prior_sigma), 0.1)
    teff_prior_sigma_log = (
        max(float(teff_prior_sigma_k) / (float(teff_prior) * math.log(10.0)), 0.02)
        if has_teff_prior
        else None
    )
    distance_pc = _distance_pc_from_candidate(candidate)
    bounds = _library_logt_bounds(library)
    teff_initial = _teff_initial_from_candidate(candidate, bounds)
    logt_initial = math.log10(teff_initial)
    lo, hi = bounds
    lo = max(float(lo), 3.2)
    hi = min(float(hi), 5.0)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        lo, hi = DEFAULT_LOGT_BOUNDS
    av_hi = float(np.clip(max(10.0, av_initial + 5.0), 10.0, 30.0))
    spectrum_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def spectrum_for(logt: float) -> tuple[np.ndarray, np.ndarray]:
        key = round(float(logt), 10)
        if key not in spectrum_cache:
            spectrum_cache[key] = _generate_spectrum(library, float(logt), logg, z)
        return spectrum_cache[key]

    def raw_band_fluxes(logt: float, av: float, frame: pd.DataFrame, *, intrinsic: bool = False) -> np.ndarray:
        wavelength, spectrum = spectrum_for(logt)
        model_spectrum = spectrum if intrinsic else apply_extinction(
            wavelength, spectrum, av, rv=DEFAULT_RV, law=EXTINCTION_LAW,
        )
        values = []
        for _, point in frame.iterrows():
            key = point["response_key"]
            try:
                values.append(
                    _model_catalog_flux_nu_jy(
                        wavelength,
                        model_spectrum,
                        responses[key],
                        point,
                    )
                )
            except Exception:
                values.append(np.nan)
        return np.asarray(values, dtype=float)

    def fit_active(frame: pd.DataFrame, starting: tuple[float, float, float] | None = None):
        obs_log = np.log10(frame["observed_flux_nu_jy"].to_numpy(dtype=float))
        sigma_log = frame["fit_sigma_log"].to_numpy(dtype=float)
        stat_sigma = frame.get("fit_sigma_log_stat", frame["fit_sigma_log"]).to_numpy(dtype=float)
        systematic_sigma = frame.get(
            "fit_sigma_log_systematic", pd.Series(0.0, index=frame.index)
        ).to_numpy(dtype=float)
        covariance = np.diag(np.square(np.clip(stat_sigma, 1.0e-6, None)))
        groups = frame.get("correlation_group", pd.Series("", index=frame.index)).fillna("").astype(str).to_numpy()
        for group in sorted(set(groups)):
            positions = np.flatnonzero(groups == group)
            if positions.size:
                covariance[np.ix_(positions, positions)] += np.outer(
                    systematic_sigma[positions], systematic_sigma[positions]
                )
        try:
            whitener = np.linalg.cholesky(covariance + np.eye(len(frame)) * 1.0e-12)
        except np.linalg.LinAlgError:
            whitener = np.diag(np.clip(sigma_log, 1.0e-6, None))

        def log_difference(params: np.ndarray) -> np.ndarray:
            logt, av_value, log_norm = (float(params[0]), float(params[1]), float(params[2]))
            try:
                raw = raw_band_fluxes(logt, av_value, frame)
            except Exception:
                return np.full(len(frame), np.nan, dtype=float)
            return obs_log - (np.log10(raw) + log_norm)

        def data_residual(params: np.ndarray) -> np.ndarray:
            difference = log_difference(params)
            if not np.all(np.isfinite(difference)):
                return np.full(len(frame), 1.0e6, dtype=float)
            return np.linalg.solve(whitener, difference)

        def point_residual(params: np.ndarray) -> np.ndarray:
            difference = log_difference(params)
            resid = difference / sigma_log
            return np.where(np.isfinite(resid), resid, 1.0e6)

        def residual(params: np.ndarray) -> np.ndarray:
            values = [*data_residual(params)]
            if has_teff_prior:
                values.append((float(params[0]) - math.log10(teff_prior)) / teff_prior_sigma_log)
            if has_av_prior:
                values.append((float(params[1]) - av_initial) / av_prior_sigma)
            return np.asarray(values, dtype=float)

        if starting is not None:
            starts: list[tuple[float, float]] = [(starting[0], starting[1])]
        else:
            alternate_teff = 10000.0 if teff_initial < 8000.0 else 5000.0
            starts = [
                (logt_initial, av_initial),
                (math.log10(6000.0), 0.0),
                (math.log10(alternate_teff), min(max(av_initial, 0.5), av_hi)),
            ]
        best = None
        for start_logt, start_av in starts:
            start_logt = float(np.clip(start_logt, lo, hi))
            start_av = float(np.clip(start_av, 0.0, av_hi))
            try:
                raw = raw_band_fluxes(start_logt, start_av, frame)
            except Exception:
                continue
            valid = np.isfinite(raw) & (raw > 0)
            if not np.all(valid):
                continue
            log_norm = float(np.average(obs_log - np.log10(raw), weights=1.0 / np.square(sigma_log)))
            x0 = np.array([start_logt, start_av, np.clip(log_norm, -70.0, -20.0)])
            try:
                result = least_squares(
                    residual,
                    x0,
                    bounds=([lo, 0.0, -70.0], [hi, av_hi, -20.0]),
                    loss="soft_l1",
                    f_scale=1.5,
                    x_scale=np.array([0.1, 1.0, 5.0]),
                    max_nfev=160,
                )
            except Exception:
                continue
            if result.success and np.isfinite(result.cost) and (best is None or result.cost < best.cost):
                best = result
        return best, point_residual, data_residual

    result, point_residual_function, whitened_residual_function = fit_active(usable)
    if result is None:
        fit_row = _empty_fit_row(candidate_id, status="fit_failed", warning="Bandpass Kurucz optimizer did not converge.", row=candidate)
        fit_row["n_available_points"] = int(len(diagnostics))
        fit_row["measurement_set_hash"] = measurement_hash
        fit_row["response_manifest_hash"] = response_hash
        fit_row["calibration_manifest_hash"] = calibration_hash
        fit_row["fit_recipe_hash"] = recipe_hash
        fit_row["fit_run_hash"] = run_hash
        return fit_row, pd.DataFrame(columns=SED_MODEL_CURVE_COLUMNS), _finalize_point_rows(diagnostics, run_hash)

    for _ in range(3):
        standardized = point_residual_function(result.x)
        if len(usable) <= MIN_FIT_POINTS or not np.any(np.isfinite(standardized)):
            break
        worst_position = int(np.nanargmax(np.abs(standardized)))
        if abs(float(standardized[worst_position])) <= OUTLIER_SIGMA:
            break
        rejected_index = usable.iloc[worst_position].name
        diagnostics.loc[rejected_index, "used"] = False
        diagnostics.loc[rejected_index, "exclusion_reason"] = "robust_outlier"
        usable = diagnostics.loc[_bool_series(diagnostics["used"])].copy()
        second, second_point_residual, second_whitened_residual = fit_active(usable, tuple(result.x))
        if second is None:
            diagnostics.loc[rejected_index, "used"] = True
            diagnostics.loc[rejected_index, "exclusion_reason"] = ""
            usable = diagnostics.loc[_bool_series(diagnostics["used"])].copy()
            break
        result = second
        point_residual_function = second_point_residual
        whitened_residual_function = second_whitened_residual

    logt_best, av_best, log_apparent_scale = map(float, result.x)
    final_residual = whitened_residual_function(result.x)
    chi2 = float(np.sum(np.square(final_residual)))
    apparent_scale = float(10.0 ** log_apparent_scale)
    teff = float(10.0 ** logt_best)
    dof = max(int(len(usable)) - 3, 1)
    reduced_chi2 = float(chi2 / dof)
    luminosity_lsun = np.nan
    if distance_pc is not None and distance_pc > 0:
        luminosity_lsun = apparent_scale * 4.0 * math.pi * (distance_pc * PC_CM) ** 2
    scale = luminosity_lsun
    radius_rsun = (
        math.sqrt(luminosity_lsun) * (5772.0 / teff) ** 2
        if math.isfinite(luminosity_lsun) and luminosity_lsun > 0 and teff > 0
        else np.nan
    )
    band_records = [
        {
            "measurement_id": str(row.get("measurement_id") or ""),
            "source": str(row.get("source") or ""),
            "band": str(row.get("band") or ""),
            "lambda_pivot_angstrom": _safe_float(row.get("lambda_pivot_angstrom")),
            "lambda_reference_angstrom": _safe_float(row.get("lambda_reference_angstrom")),
            "svo_filter_id": str(row.get("svo_filter_id") or ""),
            "response_hash": responses[row["response_key"]].response_hash,
            "calibration_id": str(row.get("calibration_id") or ""),
            "passband_fidelity": str(row.get("passband_fidelity") or ""),
        }
        for _, row in usable.iterrows()
    ]
    teff_err = np.nan
    av_err = np.nan
    covariance = np.full((3, 3), np.nan, dtype=float)
    covariance_status = "unavailable"
    try:
        covariance = np.linalg.pinv(result.jac.T @ result.jac) * reduced_chi2
        covariance = 0.5 * (covariance + covariance.T)
        if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
            raise ValueError("non-finite optimizer covariance")
        eigenvalues = np.linalg.eigvalsh(covariance)
        if float(np.nanmin(eigenvalues)) < -1.0e-12:
            raise ValueError("optimizer covariance is not positive semidefinite")
        covariance_status = "ok" if float(np.nanmin(eigenvalues)) > 1.0e-14 else "singular"
        teff_err = math.log(10.0) * teff * math.sqrt(max(float(covariance[0, 0]), 0.0))
        av_err = math.sqrt(max(float(covariance[1, 1]), 0.0))
    except Exception as exc:
        covariance = np.full((3, 3), np.nan, dtype=float)
        covariance_status = f"failed:{type(exc).__name__}"
    boundary_flags = []
    if abs(logt_best - lo) < 1.0e-3 or abs(logt_best - hi) < 1.0e-3:
        boundary_flags.append("teff_boundary")
    if av_best < 1.0e-3 or abs(av_best - av_hi) < 1.0e-3:
        boundary_flags.append("av_boundary")
    fit_row = {
        "candidate_id": str(candidate_id),
        "model_family": MODEL_FAMILY,
        "fit_version": SED_MODEL_FIT_VERSION,
        "photometry_method": "bandpass_integrated",
        "extinction_law": EXTINCTION_LAW,
        "teff_k": teff,
        "teff_err_k": teff_err,
        "logg": logg,
        "z": z,
        "av_fixed": av_initial,
        "av_fit": av_best,
        "av_err": av_err,
        "rv": DEFAULT_RV,
        "apparent_scale": apparent_scale,
        "scale": scale,
        "luminosity_lsun": luminosity_lsun,
        "radius_rsun": radius_rsun,
        "chi2": float(chi2),
        "reduced_chi2": reduced_chi2,
        "n_fit_points": int(len(usable)),
        "n_available_points": int(len(diagnostics)),
        "n_rejected_points": int((diagnostics["exclusion_reason"].astype(str) == "robust_outlier").sum()),
        "fit_lambda_min": float(np.nanmin(usable["fit_wavelength_angstrom"])),
        "fit_lambda_max": float(np.nanmax(usable["fit_wavelength_angstrom"])),
        "fit_bands_json": json.dumps(band_records, separators=(",", ":")),
        "priors_json": json.dumps({
            "teff_k": teff_prior if has_teff_prior else None,
            "teff_sigma_k": teff_prior_sigma_k if has_teff_prior else None,
            "teff_sigma_log10": teff_prior_sigma_log if has_teff_prior else None,
            "av": av_initial if has_av_prior else None,
            "av_sigma": av_prior_sigma if has_av_prior else None,
        }, separators=(",", ":")),
        "fit_param_names_json": json.dumps(
            ["log10_teff", "av", "log10_apparent_scale"], separators=(",", ":")
        ),
        "fit_param_values_json": json.dumps(
            [logt_best, av_best, log_apparent_scale], separators=(",", ":")
        ),
        "fit_covariance_json": (
            json.dumps(covariance.tolist(), separators=(",", ":"))
            if np.all(np.isfinite(covariance))
            else "[]"
        ),
        "fit_covariance_status": covariance_status,
        "robust_objective": float(2.0 * result.cost),
        "measurement_set_hash": measurement_hash,
        "response_manifest_hash": response_hash,
        "calibration_manifest_hash": calibration_hash,
        "fit_recipe_hash": recipe_hash,
        "fit_run_hash": run_hash,
        "boundary_flags": ";".join(boundary_flags),
        "status": "ok",
        "warning": "",
    }

    native_wavelength, native_spectrum = spectrum_for(logt_best)
    requested_response_max = max(
        (
            float(responses[key].wavelength_angstrom[-1])
            for key in diagnostics.get("response_key", pd.Series(dtype=object)).dropna()
            if key in responses
        ),
        default=float(native_wavelength[-1]),
    )
    wavelength, spectrum = _extend_stellar_spectrum_rayleigh_jeans(
        native_wavelength,
        native_spectrum,
        requested_response_max,
    )
    # Keep the likelihood restricted to the optical points selected above, but
    # retain the full photosphere curve so IR excesses are visible against it.
    curve_min = float(np.nanmin(wavelength))
    curve_max = float(np.nanmax(wavelength))
    if curve_min >= curve_max:
        curve_wave = wavelength
    else:
        curve_wave = np.geomspace(curve_min, curve_max, max(int(curve_points), 32))
        if float(wavelength[-1]) > float(native_wavelength[-1]):
            curve_wave = np.union1d(
                curve_wave,
                np.asarray([native_wavelength[-1], wavelength[-1]], dtype=float),
            )
    intrinsic_spectrum = np.interp(curve_wave, wavelength, spectrum, left=np.nan, right=np.nan) * apparent_scale
    extincted_spectrum = apply_extinction(
        wavelength,
        spectrum,
        av_best,
        rv=DEFAULT_RV,
        law=EXTINCTION_LAW,
    )
    observed_spectrum = np.interp(
        curve_wave,
        wavelength,
        extincted_spectrum,
        left=np.nan,
        right=np.nan,
    ) * apparent_scale
    if distance_pc is not None and distance_pc > 0:
        factor = 4.0 * math.pi * (distance_pc * PC_CM) ** 2
        intrinsic_l_lambda = factor * curve_wave * intrinsic_spectrum
        observed_l_lambda = factor * curve_wave * observed_spectrum
    else:
        intrinsic_l_lambda = np.full_like(curve_wave, np.nan, dtype=float)
        observed_l_lambda = np.full_like(curve_wave, np.nan, dtype=float)
    curve = pd.DataFrame({
        "candidate_id": str(candidate_id),
        "model_family": MODEL_FAMILY,
        "fit_version": SED_MODEL_FIT_VERSION,
        "fit_run_hash": run_hash,
        "wavelength_angstrom": curve_wave,
        "lambda_l_lambda": intrinsic_l_lambda,
        "flux_lambda": intrinsic_spectrum,
        "lambda_l_lambda_intrinsic": intrinsic_l_lambda,
        "lambda_l_lambda_observed": observed_l_lambda,
        "flux_lambda_intrinsic": intrinsic_spectrum,
        "flux_lambda_observed": observed_spectrum,
        "teff_k": teff,
        "av_fit": av_best,
        "scale": scale,
    })
    curve = curve[np.isfinite(curve["wavelength_angstrom"]) & np.isfinite(curve["flux_lambda_intrinsic"]) & (curve["flux_lambda_intrinsic"] > 0)]

    diagnostics["used"] = diagnostics.index.isin(usable.index)
    intrinsic_native = spectrum
    observed_native = extincted_spectrum
    for idx, row in diagnostics.iterrows():
        key = row.get("response_key")
        response = responses.get(key)
        if response is None:
            continue
        try:
            intrinsic_fnu = apparent_scale * _model_catalog_flux_nu_jy(
                wavelength, intrinsic_native, response, row
            )
            observed_fnu = apparent_scale * _model_catalog_flux_nu_jy(
                wavelength, observed_native, response, row
            )
        except BandpassUnavailableError as exc:
            diagnostics.at[idx, "prediction_status"] = "unavailable"
            diagnostics.at[idx, "prediction_reason"] = f"model_wavelength_coverage:{exc}"
            diagnostics.at[idx, "used"] = False
            continue
        diagnostics.at[idx, "prediction_status"] = "ok"
        diagnostics.at[idx, "prediction_reason"] = ""
        diagnostics.at[idx, "model_flux_nu_jy"] = observed_fnu
        diagnostics.at[idx, "model_flux_nu_jy_intrinsic"] = intrinsic_fnu
        diagnostics.at[idx, "response_hash"] = response.response_hash
        lam = _display_wavelength(row)
        if lam is not None and lam > 0:
            observed_flam = _flam_from_fnu(observed_fnu, lam)
            intrinsic_flam = _flam_from_fnu(intrinsic_fnu, lam)
            diagnostics.at[idx, "model_flux_lambda"] = observed_flam
            diagnostics.at[idx, "model_flux_lambda_intrinsic"] = intrinsic_flam
            if distance_pc is not None and distance_pc > 0:
                factor = 4.0 * math.pi * (distance_pc * PC_CM) ** 2 * lam
                diagnostics.at[idx, "model_lambda_l_lambda"] = factor * observed_flam
                diagnostics.at[idx, "model_lambda_l_lambda_intrinsic"] = factor * intrinsic_flam
        observed_fnu_value = _safe_float(row.get("observed_flux_nu_jy"))
        sigma_value = _safe_float(row.get("fit_sigma_log"))
        if observed_fnu_value is not None and observed_fnu_value > 0 and sigma_value is not None and sigma_value > 0:
            diagnostics.at[idx, "residual_sigma"] = (math.log10(observed_fnu_value) - math.log10(observed_fnu)) / sigma_value
        calibration = _calibration_for_row(row, response)
        zero_point = _safe_float(getattr(calibration, "zero_point_jy", None))
        if zero_point is not None and zero_point > 0 and observed_fnu > 0:
            diagnostics.at[idx, "model_mag"] = -2.5 * math.log10(observed_fnu / zero_point)

    for column in SED_MODEL_CURVE_COLUMNS:
        if column not in curve.columns:
            curve[column] = None
    return fit_row, curve[SED_MODEL_CURVE_COLUMNS].reset_index(drop=True), _finalize_point_rows(diagnostics, run_hash)


def _finalize_point_rows(frame: pd.DataFrame, run_hash: str = "") -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=SED_MODEL_POINT_COLUMNS)
    out = frame.copy()
    out["fit_version"] = SED_MODEL_FIT_VERSION
    out["fit_run_hash"] = str(run_hash or "")
    pending = out.get("prediction_status", pd.Series("", index=out.index)).fillna("").astype(str).isin({"", "pending"})
    out.loc[pending, "prediction_status"] = "not_generated"
    out.loc[pending & (out.get("prediction_reason", "") == ""), "prediction_reason"] = "fit_unavailable"
    rename = {
        "mag_system": "mag_system",
        "svo_filter_id": "svo_filter_id",
        "flux_lambda": "observed_flux_lambda",
        "flux_lambda_err": "observed_flux_lambda_err",
        "lambda_l_lambda": "observed_lambda_l_lambda",
        "lambda_l_lambda_err": "observed_lambda_l_lambda_err",
    }
    out = out.rename(columns=rename)
    for col in SED_MODEL_POINT_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out["candidate_id"] = out["candidate_id"].astype(str)
    out["used"] = _bool_series(out["used"]).astype(int)
    return out[SED_MODEL_POINT_COLUMNS].reset_index(drop=True)


def fit_sed_models(
    candidates: pd.DataFrame,
    sed_rows: pd.DataFrame,
    *,
    library: object | None = None,
    curve_points: int = 400,
    progress_callback: Callable[[str], None] | None = None,
    response_loader: ResponseLoader | None = None,
    allow_bandpass_download: bool = True,
    return_points: bool = False,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit Castelli/Kurucz models using extinction-aware synthetic photometry."""
    library = library if library is not None else _load_kurucz_library()
    model_grid_provenance = sed_model_grid_provenance(library)
    model_grid_hash = hashlib.sha256(
        json.dumps(
            model_grid_provenance,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    recipe_hash = sed_fit_recipe_hash(library)
    candidates = pd.DataFrame() if candidates is None else candidates.copy()
    sed_rows = pd.DataFrame() if sed_rows is None else sed_rows.copy()

    if not sed_rows.empty:
        _canonicalize_response_fields(sed_rows)

    filter_pairs = []
    if not sed_rows.empty:
        filter_pairs = [
            _canonical_response_key(row) for _, row in sed_rows.iterrows()
        ]
    responses, response_failures = build_response_map(
        filter_pairs,
        allow_download=allow_bandpass_download,
        progress_callback=progress_callback,
        response_loader=response_loader,
    )
    if progress_callback and response_failures:
        progress_callback(
            f"[SED bandpass] {len(responses)} response(s) ready; "
            f"{len(response_failures)} unavailable"
        )

    candidate_map: dict[str, pd.Series | dict] = {}
    if not candidates.empty:
        for idx, row in candidates.iterrows():
            cid = _candidate_id_for_row(row, idx)
            candidate_map[str(cid)] = row
    if not sed_rows.empty and "candidate_id" in sed_rows.columns:
        for cid in sed_rows["candidate_id"].dropna().astype(str).unique():
            candidate_map.setdefault(str(cid), {"candidate_id": str(cid)})

    def fit_one(item: tuple[str, pd.Series | dict]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
        candidate_id, candidate = item
        measurement_hash = ""
        candidate_hash = sed_candidate_context_hash(candidate)
        candidate_points = pd.DataFrame()
        candidate_response_hash = ""
        calibration_hash = ""
        run_hash = ""
        try:
            candidate_points = _prepare_candidate_points(
                candidate_id,
                candidate,
                sed_rows,
                responses,
                response_failures,
            )
            measurement_hash = hash_sed_measurement_set(candidate_points)
            candidate_response_hash = _candidate_response_manifest_hash(candidate_points, responses)
            calibration_hash = _candidate_calibration_manifest_hash(candidate_points)
            input_policy_hash = _candidate_input_policy_manifest_hash(candidate_points)
            run_hash = _fit_run_hash(
                candidate_id,
                measurement_hash,
                candidate_hash,
                candidate_response_hash,
                calibration_hash,
                input_policy_hash,
                recipe_hash,
                model_grid_hash,
            )
            fit_row, curve, point_rows = _fit_single_candidate(
                candidate_id,
                candidate,
                candidate_points,
                library,
                responses,
                measurement_hash,
                candidate_response_hash,
                calibration_hash,
                recipe_hash,
                run_hash,
                curve_points=curve_points,
            )
            final_policy_hash = _candidate_input_policy_manifest_hash(point_rows)
            run_hash = _fit_run_hash(
                candidate_id,
                measurement_hash,
                candidate_hash,
                candidate_response_hash,
                calibration_hash,
                final_policy_hash,
                recipe_hash,
                model_grid_hash,
            )
            fit_row["measurement_set_hash"] = measurement_hash
            fit_row["input_policy_manifest_hash"] = final_policy_hash
            fit_row["fit_run_hash"] = run_hash
            fit_row["candidate_context_hash"] = candidate_hash
            fit_row["model_grid_hash"] = model_grid_hash
            fit_row["model_grid_provenance_json"] = json.dumps(
                model_grid_provenance,
                sort_keys=True,
                separators=(",", ":"),
            )
            if isinstance(curve, pd.DataFrame) and not curve.empty:
                curve["fit_run_hash"] = run_hash
            if isinstance(point_rows, pd.DataFrame) and not point_rows.empty:
                point_rows["fit_run_hash"] = run_hash
                point_rows["input_hash"] = [
                    make_sed_input_hash(row) for _, row in point_rows.iterrows()
                ]
            return fit_row, curve, point_rows
        except PystellibsSetupError:
            raise
        except Exception as exc:
            if not candidate_points.empty:
                measurement_hash = hash_sed_measurement_set(candidate_points)
                input_policy_hash = _candidate_input_policy_manifest_hash(candidate_points)
                run_hash = _fit_run_hash(
                    candidate_id,
                    measurement_hash,
                    candidate_hash,
                    candidate_response_hash,
                    calibration_hash,
                    input_policy_hash,
                    recipe_hash,
                    model_grid_hash,
                )
            else:
                input_policy_hash = ""
            fit_row = _empty_fit_row(candidate_id, status="fit_failed", warning=str(exc), row=candidate)
            fit_row["measurement_set_hash"] = measurement_hash
            fit_row["candidate_context_hash"] = candidate_hash
            fit_row["response_manifest_hash"] = candidate_response_hash
            fit_row["calibration_manifest_hash"] = calibration_hash
            fit_row["input_policy_manifest_hash"] = input_policy_hash
            fit_row["fit_recipe_hash"] = recipe_hash
            fit_row["model_grid_hash"] = model_grid_hash
            fit_row["model_grid_provenance_json"] = json.dumps(
                model_grid_provenance,
                sort_keys=True,
                separators=(",", ":"),
            )
            fit_row["fit_run_hash"] = run_hash
            curve = pd.DataFrame(columns=SED_MODEL_CURVE_COLUMNS)
            point_rows = _finalize_point_rows(candidate_points, run_hash)
            if not point_rows.empty:
                point_rows["input_hash"] = [
                    make_sed_input_hash(row) for _, row in point_rows.iterrows()
                ]
            return fit_row, curve, point_rows

    fit_rows: list[dict] = []
    curve_parts: list[pd.DataFrame] = []
    point_parts: list[pd.DataFrame] = []
    items = list(candidate_map.items())
    total = len(items)
    worker_count = max(int(workers), 1)
    if worker_count == 1 or total <= 1:
        result_iter = map(fit_one, items)
    else:
        executor = ThreadPoolExecutor(max_workers=min(worker_count, total), thread_name_prefix="sed-fit")
        result_iter = executor.map(fit_one, items)
    try:
        for idx, ((candidate_id, _candidate), result_item) in enumerate(zip(items, result_iter, strict=False), start=1):
            if progress_callback and (idx == 1 or idx % 50 == 0 or idx == total):
                progress_callback(f"[SED model] fitted {idx}/{total}: {candidate_id}")
            fit_row, curve, point_rows = result_item
            fit_rows.append(fit_row)
            if isinstance(curve, pd.DataFrame) and not curve.empty:
                curve_parts.append(curve)
            if isinstance(point_rows, pd.DataFrame) and not point_rows.empty:
                point_parts.append(point_rows)
    finally:
        if worker_count > 1 and total > 1:
            executor.shutdown(wait=True)

    fits = pd.DataFrame(fit_rows, columns=SED_MODEL_FIT_COLUMNS)
    if curve_parts:
        curves = pd.concat(curve_parts, ignore_index=True)
    else:
        curves = pd.DataFrame(columns=SED_MODEL_CURVE_COLUMNS)
    points = pd.concat(point_parts, ignore_index=True) if point_parts else pd.DataFrame(columns=SED_MODEL_POINT_COLUMNS)
    for col in SED_MODEL_FIT_COLUMNS:
        if col not in fits.columns:
            fits[col] = None
    for col in SED_MODEL_CURVE_COLUMNS:
        if col not in curves.columns:
            curves[col] = None
    for col in SED_MODEL_POINT_COLUMNS:
        if col not in points.columns:
            points[col] = None
    if return_points:
        return fits[SED_MODEL_FIT_COLUMNS], curves[SED_MODEL_CURVE_COLUMNS], points[SED_MODEL_POINT_COLUMNS]
    return fits[SED_MODEL_FIT_COLUMNS], curves[SED_MODEL_CURVE_COLUMNS]


def load_sed_model_fits(conn: sqlite3.Connection, candidate_id: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(
            f"SELECT {', '.join(SED_MODEL_FIT_COLUMNS)} FROM {SED_MODEL_FIT_TABLE_NAME} WHERE candidate_id = ?",
            conn,
            params=(str(candidate_id),),
        )
    except Exception:
        return pd.DataFrame(columns=SED_MODEL_FIT_COLUMNS)


def load_sed_model_curves(conn: sqlite3.Connection, candidate_id: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(
            f"SELECT {', '.join(SED_MODEL_CURVE_COLUMNS)} FROM {SED_MODEL_CURVE_TABLE_NAME} WHERE candidate_id = ? ORDER BY wavelength_angstrom",
            conn,
            params=(str(candidate_id),),
        )
    except Exception:
        return pd.DataFrame(columns=SED_MODEL_CURVE_COLUMNS)


def load_sed_model_points(conn: sqlite3.Connection, candidate_id: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(
            f"SELECT {', '.join(SED_MODEL_POINT_COLUMNS)} FROM {SED_MODEL_POINT_TABLE_NAME} "
            "WHERE candidate_id = ? ORDER BY lambda_eff_angstrom, source, band",
            conn,
            params=(str(candidate_id),),
        )
    except Exception:
        return pd.DataFrame(columns=SED_MODEL_POINT_COLUMNS)


def _sqlite_value(value: object) -> object:
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        if value is None:
            return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def upsert_sed_model_fits(
    conn: sqlite3.Connection,
    fits: pd.DataFrame,
    *,
    commit: bool = True,
) -> int:
    if fits is None or fits.empty:
        return 0
    frame = fits.copy()
    for col in SED_MODEL_FIT_COLUMNS:
        if col not in frame.columns:
            frame[col] = None
    frame = frame[SED_MODEL_FIT_COLUMNS]
    placeholders = ", ".join(["?"] * len(SED_MODEL_FIT_COLUMNS))
    assignments = ", ".join([f"{col}=excluded.{col}" for col in SED_MODEL_FIT_COLUMNS if col != "candidate_id"])
    sql = (
        f"INSERT INTO {SED_MODEL_FIT_TABLE_NAME} ({', '.join(SED_MODEL_FIT_COLUMNS)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT(candidate_id) DO UPDATE SET {assignments}"
    )
    count = 0
    for _, row in frame.iterrows():
        conn.execute(sql, [_sqlite_value(row[col]) for col in SED_MODEL_FIT_COLUMNS])
        count += 1
    if commit:
        conn.commit()
    return count


def upsert_sed_model_curves(
    conn: sqlite3.Connection,
    curves: pd.DataFrame,
    *,
    replace_candidate_ids: Iterable[str] | None = None,
    commit: bool = True,
) -> int:
    candidate_ids = [str(cid) for cid in (replace_candidate_ids or [])]
    if curves is not None and not curves.empty and "candidate_id" in curves.columns:
        candidate_ids.extend(str(cid) for cid in curves["candidate_id"].dropna().astype(str).unique())
    for cid in sorted(set(candidate_ids)):
        conn.execute(f"DELETE FROM {SED_MODEL_CURVE_TABLE_NAME} WHERE candidate_id = ?", (cid,))

    if curves is None or curves.empty:
        if commit:
            conn.commit()
        return 0
    frame = curves.copy()
    for col in SED_MODEL_CURVE_COLUMNS:
        if col not in frame.columns:
            frame[col] = None
    frame = frame[SED_MODEL_CURVE_COLUMNS]
    placeholders = ", ".join(["?"] * len(SED_MODEL_CURVE_COLUMNS))
    sql = f"INSERT INTO {SED_MODEL_CURVE_TABLE_NAME} ({', '.join(SED_MODEL_CURVE_COLUMNS)}) VALUES ({placeholders})"
    count = 0
    for _, row in frame.iterrows():
        conn.execute(sql, [_sqlite_value(row[col]) for col in SED_MODEL_CURVE_COLUMNS])
        count += 1
    if commit:
        conn.commit()
    return count


def upsert_sed_model_points(
    conn: sqlite3.Connection,
    points: pd.DataFrame,
    *,
    replace_candidate_ids: Iterable[str] | None = None,
    commit: bool = True,
) -> int:
    candidate_ids = [str(cid) for cid in (replace_candidate_ids or [])]
    if points is not None and not points.empty and "candidate_id" in points.columns:
        candidate_ids.extend(str(cid) for cid in points["candidate_id"].dropna().astype(str).unique())
    for cid in sorted(set(candidate_ids)):
        conn.execute(f"DELETE FROM {SED_MODEL_POINT_TABLE_NAME} WHERE candidate_id = ?", (cid,))

    if points is None or points.empty:
        if commit:
            conn.commit()
        return 0
    frame = points.copy()
    for col in SED_MODEL_POINT_COLUMNS:
        if col not in frame.columns:
            frame[col] = None
    frame = frame[SED_MODEL_POINT_COLUMNS]
    placeholders = ", ".join(["?"] * len(SED_MODEL_POINT_COLUMNS))
    sql = f"INSERT INTO {SED_MODEL_POINT_TABLE_NAME} ({', '.join(SED_MODEL_POINT_COLUMNS)}) VALUES ({placeholders})"
    count = 0
    for _, row in frame.iterrows():
        conn.execute(sql, [_sqlite_value(row[col]) for col in SED_MODEL_POINT_COLUMNS])
        count += 1
    if commit:
        conn.commit()
    return count


def upsert_sed_model_results(
    conn: sqlite3.Connection,
    fits: pd.DataFrame,
    curves: pd.DataFrame,
    points: pd.DataFrame | None = None,
    *,
    replace_candidate_ids: Iterable[str] | None = None,
    measurement_rows: pd.DataFrame | None = None,
) -> tuple[int, int]:
    fits = pd.DataFrame() if fits is None else fits.copy()
    curves = pd.DataFrame() if curves is None else curves.copy()
    points = None if points is None else points.copy()
    for frame in (fits, curves, points):
        if frame is not None and "fit_run_id" in frame.columns:
            frame["fit_run_id"] = frame["fit_run_id"].astype(object)
    run_ids: dict[str, str] = {}
    if not fits.empty:
        for idx, row in fits.iterrows():
            if _row_text(row, "fit_run_hash"):
                fit_run_id = make_sed_result_fit_run_id(row.to_dict())
                fits.at[idx, "fit_run_id"] = fit_run_id
                run_ids[str(row.get("candidate_id"))] = fit_run_id
    if points is not None and not points.empty:
        for idx, row in points.iterrows():
            candidate_id = str(row.get("candidate_id"))
            if candidate_id in run_ids:
                points.at[idx, "fit_run_id"] = run_ids[candidate_id]
            points.at[idx, "input_hash"] = make_sed_input_hash(points.loc[idx])
    if not curves.empty:
        for idx, row in curves.iterrows():
            candidate_id = str(row.get("candidate_id"))
            if candidate_id in run_ids:
                curves.at[idx, "fit_run_id"] = run_ids[candidate_id]
    candidate_ids = replace_candidate_ids
    if candidate_ids is None and fits is not None and not fits.empty and "candidate_id" in fits.columns:
        candidate_ids = fits["candidate_id"].dropna().astype(str).unique().tolist()
    savepoint = "sed_model_results"
    conn.execute(f"SAVEPOINT {savepoint}")
    try:
        has_v3_storage = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sed_fit_runs'"
        ).fetchone() is not None
        if has_v3_storage and measurement_rows is not None and not measurement_rows.empty:
            from malca.review.sed_storage import store_canonical_sed_rows

            store_canonical_sed_rows(conn, measurement_rows, commit=False)
        replacement_ids = sorted({str(value) for value in (candidate_ids or [])})
        incoming_fit_ids = (
            set(fits["candidate_id"].dropna().astype(str))
            if fits is not None and not fits.empty and "candidate_id" in fits.columns
            else set()
        )
        for candidate_id in replacement_ids:
            if candidate_id not in incoming_fit_ids:
                conn.execute(
                    f"DELETE FROM {SED_MODEL_FIT_TABLE_NAME} WHERE candidate_id = ?",
                    (candidate_id,),
                )
        n_fits = upsert_sed_model_fits(conn, fits, commit=False)
        n_curves = upsert_sed_model_curves(
            conn, curves, replace_candidate_ids=candidate_ids, commit=False
        )
        if points is not None:
            upsert_sed_model_points(
                conn, points, replace_candidate_ids=candidate_ids, commit=False
            )
        if has_v3_storage:
            from malca.review.sed_storage import store_sed_fit_results

            store_sed_fit_results(conn, fits, points, commit=False)
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
    except Exception:
        conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        raise
    conn.commit()
    return n_fits, n_curves
