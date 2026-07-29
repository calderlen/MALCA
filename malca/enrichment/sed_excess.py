"""Model-aware infrared SED excess posteriors and classifications.

The excess calculation is performed in observed ``F_nu`` space.  Distance is
deliberately absent: it cancels in the observed/photosphere flux ratio.  The
photosphere posterior is the correlated CK04 fit posterior in
``(log10 Teff, Av, log10 apparent scale)``.  WISE measurement, calibration,
variability, and shared non-simultaneity terms are then added in log-flux
space.  An optional empirical null locus absorbs residual atmosphere/catalog
systematics measured from clean control stars.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm

from malca.enrichment.sed_model import (
    DEFAULT_RV,
    EXTINCTION_LAW,
    LSUN_ERG_S,
    SED_MODEL_FIT_VERSION,
    _generate_spectrum,
    _library_logt_bounds,
    _load_kurucz_library,
    _prepare_candidate_points,
    _to_bool,
)
from malca.enrichment.synthetic_photometry import (
    FilterResponse,
    ResponseLoader,
    apply_extinction,
    bandpass_flux_nu_jy,
    build_response_map,
)
from malca.review.sed import bandpass_for


SED_EXCESS_VERSION = "wise-mc-v1"
WISE_BANDS = ("W1", "W2", "W3", "W4")
WISE_BAND_INDEX = {band: index for index, band in enumerate(WISE_BANDS)}
WISE_CALIBRATION_FLOOR_MAG = {"W1": 0.03, "W2": 0.03, "W3": 0.05, "W4": 0.10}
WISE_MIN_SNR = 3.0
WISE_MAX_RCHI2 = 3.0
WISE_MAX_SEP_ARCSEC = 2.0
DEFAULT_NON_SIMULTANEITY_FLOOR_MAG = 0.03
MODEL_PREDICTION_METHOD = "ck04-bandpass-grid-loginterp-v1"
DIRECT_MODEL_PREDICTION_METHOD = "ck04-bandpass-direct-v1"
PRIMARY_EXCESS_PROBABILITY = 0.997
ADJACENT_SUPPORT_PROBABILITY = 0.95
PROBABLE_EXCESS_PROBABILITY = 0.95

SED_EXCESS_BAND_COLUMNS = [
    "candidate_id", "excess_version", "fit_version", "fit_run_hash", "source", "band",
    "observed_flux_nu_jy", "observed_flux_nu_jy_err", "observed_mag", "observed_mag_err",
    "model_flux_nu_jy_p16", "model_flux_nu_jy_p50", "model_flux_nu_jy_p84",
    "ratio_p16", "ratio_p50", "ratio_p84",
    "excess_fraction_p16", "excess_fraction_p50", "excess_fraction_p84",
    "log_ratio_p16", "log_ratio_p50", "log_ratio_p84", "p_excess", "z_excess",
    "null_offset_dex", "null_scatter_dex",
    "calibrated_ratio_p16", "calibrated_ratio_p50", "calibrated_ratio_p84",
    "p_excess_calibrated", "z_excess_calibrated",
    "measurement_sigma_mag", "calibration_sigma_mag", "variability_sigma_mag",
    "non_simultaneity_sigma_mag", "total_observed_sigma_mag",
    "quality_pass", "quality_status", "quality_flags", "quality_reasons",
    "posterior_reliable", "posterior_status", "posterior_flags", "posterior_acceptance_fraction",
    "n_draws", "random_seed", "null_locus_version", "model_prediction_method",
]

SED_EXCESS_SUMMARY_COLUMNS = [
    "candidate_id", "excess_version", "excess_class", "classification_reason",
    "primary_band", "adjacent_support", "posterior_reliable", "quality_summary",
    "w3_ratio_p50", "w3_ratio_p16", "w3_ratio_p84", "w3_p_excess_calibrated",
    "w4_ratio_p50", "w4_ratio_p16", "w4_ratio_p84", "w4_p_excess_calibrated",
    "sed_alpha", "sed_alpha_class",
    "dust_model", "dust_fit_status", "dust_temperature_k_p16", "dust_temperature_k_p50",
    "dust_temperature_k_p84", "lir_lstar_p16", "lir_lstar_p50", "lir_lstar_p84",
]

NULL_LOCUS_COLUMNS = [
    "excess_version", "null_locus_version", "band", "n_control", "status",
    "feature_names_json", "feature_centers_json", "feature_scales_json", "coefficients_json",
    "scatter_dex", "clip_fraction",
]


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_default(value: object) -> object:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _json_array(value: object) -> list:
    if isinstance(value, list):
        return value
    try:
        decoded = json.loads(str(value or "[]"))
    except Exception:
        return []
    return decoded if isinstance(decoded, list) else []


def _stable_seed(candidate_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{candidate_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def _quality_mapping(value: object) -> dict[str, str]:
    text = str(value or "").strip()
    out: dict[str, str] = {}
    for token in re.split(r"[;|]", text):
        if "=" not in token:
            continue
        key, item = token.split("=", 1)
        out[key.strip().lower()] = item.strip()
    return out


def _band_character(value: object, band: str) -> str:
    text = str(value or "").strip()
    index = WISE_BAND_INDEX[band]
    return text[index] if len(text) > index else ""


def _candidate_wise_quality_flags(candidate: Mapping[str, object], band: str) -> str:
    number = WISE_BAND_INDEX[band] + 1
    parts = []
    for output_name, payload_name in (
        ("qph", "allwise_ph_qual"), ("ccf", "allwise_cc_flags"),
        ("ex", "allwise_ext_flg"), ("nb", "allwise_nb"), ("na", "allwise_na"),
        ("var", "allwise_var_flg"), (f"snr{number}", f"allwise_w{number}_snr"),
        (f"chi2w{number}", f"allwise_w{number}_rchi2"),
        (f"sat{number}", f"allwise_w{number}_sat"),
        (f"nw{number}", f"allwise_w{number}_ndet"),
        (f"mw{number}", f"allwise_w{number}_nframe"),
    ):
        value = candidate.get(payload_name)
        try:
            missing = value is None or pd.isna(value)
        except (TypeError, ValueError):
            missing = value is None
        if not missing and str(value).strip() not in {"", "nan", "--"}:
            parts.append(f"{output_name}={str(value).strip()}")
    return ";".join(parts)


def evaluate_wise_quality(
    row: Mapping[str, object],
    band: str,
    *,
    candidate: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return strict, band-specific AllWISE quality status and reasons."""

    band = str(band).upper()
    if band not in WISE_BAND_INDEX:
        return {"quality_pass": False, "quality_status": "unsupported", "quality_reasons": "unsupported_band"}
    flags = str(row.get("quality_flags") or "")
    if candidate:
        candidate_flags = _candidate_wise_quality_flags(candidate, band)
        flags = ";".join(part for part in (flags, candidate_flags) if part)
    values = _quality_mapping(flags)
    number = WISE_BAND_INDEX[band] + 1
    reasons: list[str] = []
    known = 0

    upper_limit = _to_bool(row.get("is_upper_limit"))
    qph = _band_character(values.get("qph", ""), band).upper()
    if qph:
        known += 1
        upper_limit = upper_limit or qph == "U"
        if qph not in {"A", "B"}:
            reasons.append(f"ph_qual_{qph.lower()}")
    ccf = _band_character(values.get("ccf", ""), band)
    if ccf:
        known += 1
        if ccf != "0":
            reasons.append(f"artifact_{ccf.lower()}")
    for key, good, reason in (("ex", 0.0, "extended_source"), ("na", 0.0, "active_deblend")):
        value = _safe_float(values.get(key))
        if value is not None:
            known += 1
            if value != good:
                reasons.append(reason)
    nb = _safe_float(values.get("nb"))
    if nb is not None:
        known += 1
        if nb > 1:
            reasons.append("multiple_psf_components")
    snr = _safe_float(values.get(f"snr{number}"))
    if snr is None:
        observed_flux = _safe_float(row.get("observed_flux_nu_jy") or row.get("flux_nu_jy"))
        observed_error = _safe_float(row.get("observed_flux_nu_jy_err") or row.get("flux_nu_jy_err"))
        magnitude_error = _safe_float(row.get("mag_err"))
        if observed_flux is not None and observed_error is not None and observed_error > 0:
            snr = observed_flux / observed_error
        elif magnitude_error is not None and magnitude_error > 0:
            snr = 1.0857362047581294 / magnitude_error
    if snr is not None:
        known += 1
        if snr < WISE_MIN_SNR:
            reasons.append("low_snr")
    rchi2 = _safe_float(values.get(f"chi2w{number}") or values.get(f"chi2W{number}".lower()))
    if rchi2 is not None:
        known += 1
        if rchi2 > WISE_MAX_RCHI2:
            reasons.append("poor_profile_fit")
    sat = _safe_float(values.get(f"sat{number}"))
    if sat is not None:
        known += 1
        if sat > 0:
            reasons.append("saturated_pixels")
    sep = _safe_float(row.get("sep_arcsec"))
    if sep is None and candidate:
        sep = _safe_float(candidate.get("allwise_sep_arcsec"))
    if sep is not None:
        known += 1
        if sep > WISE_MAX_SEP_ARCSEC:
            reasons.append("large_match_separation")
    if upper_limit:
        reasons.append("upper_limit")
        status = "upper_limit"
    elif reasons:
        status = "fail"
    elif not qph or not ccf:
        status = "unknown"
    else:
        status = "pass"
    return {
        "quality_pass": status == "pass",
        "quality_status": status,
        "quality_reasons": ";".join(sorted(set(reasons))),
        "quality_flags": flags,
    }


def parse_fit_posterior(fit: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray, str]:
    names = _json_array(fit.get("fit_param_names_json"))
    values = _json_array(fit.get("fit_param_values_json"))
    covariance = _json_array(fit.get("fit_covariance_json"))
    expected = ["log10_teff", "av", "log10_apparent_scale"]
    if names != expected or len(values) != 3:
        teff = _safe_float(fit.get("teff_k"))
        av = _safe_float(fit.get("av_fit"))
        scale = _safe_float(fit.get("apparent_scale"))
        if teff is None or teff <= 0 or av is None or scale is None or scale <= 0:
            return np.full(3, np.nan), np.full((3, 3), np.nan), "missing_parameter_center"
        values = [math.log10(teff), av, math.log10(scale)]
    center = np.asarray(values, dtype=float)
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return center, np.full((3, 3), np.nan), "missing_covariance"
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if float(np.nanmin(eigenvalues)) < -1.0e-12:
        return center, matrix, "non_psd_covariance"
    return center, matrix, "ok" if float(np.nanmin(eigenvalues)) > 1.0e-14 else "singular_covariance"


def posterior_reliability(fit: Mapping[str, object], covariance_status: str) -> tuple[bool, list[str]]:
    flags: list[str] = []
    if str(fit.get("status") or "") != "ok":
        flags.append("fit_not_ok")
    if covariance_status != "ok":
        flags.append(covariance_status)
    if str(fit.get("boundary_flags") or "").strip():
        flags.append("fit_boundary")
    reduced_chi2 = _safe_float(fit.get("reduced_chi2"))
    if reduced_chi2 is not None and reduced_chi2 > 5.0:
        flags.append("poor_reduced_chi2")
    teff = _safe_float(fit.get("teff_k"))
    teff_err = _safe_float(fit.get("teff_err_k"))
    if teff is not None and teff_err is not None and teff > 0 and teff_err / teff > 0.25:
        flags.append("large_teff_uncertainty")
    if teff is not None and teff < 3500.0:
        flags.append("cool_star_ck04_limited")
    return not flags, flags


def draw_fit_parameters(
    fit: Mapping[str, object], n_draws: int, *, seed: int, logt_bounds: tuple[float, float]
) -> tuple[np.ndarray, dict[str, object]]:
    center, covariance, covariance_status = parse_fit_posterior(fit)
    reliable, flags = posterior_reliability(fit, covariance_status)
    if covariance_status not in {"ok", "singular_covariance"}:
        return np.empty((0, 3)), {
            "posterior_reliable": False, "posterior_status": covariance_status,
            "posterior_flags": ";".join(flags), "posterior_acceptance_fraction": 0.0,
        }
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = (eigenvectors * np.clip(eigenvalues, 0.0, None)) @ eigenvectors.T
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    generated = 0
    target = max(int(n_draws), 1)
    for _ in range(12):
        batch_size = max(target - sum(len(item) for item in accepted), target // 2, 32)
        batch = rng.multivariate_normal(center, covariance, size=batch_size, check_valid="ignore")
        generated += len(batch)
        valid = (
            np.isfinite(batch).all(axis=1)
            & (batch[:, 0] >= logt_bounds[0]) & (batch[:, 0] <= logt_bounds[1])
            & (batch[:, 1] >= 0.0) & (batch[:, 1] <= 30.0)
            & (batch[:, 2] >= -70.0) & (batch[:, 2] <= -20.0)
        )
        accepted.append(batch[valid])
        if sum(len(item) for item in accepted) >= target:
            break
    draws = np.concatenate(accepted, axis=0)[:target] if accepted else np.empty((0, 3))
    acceptance = len(draws) / max(generated, 1)
    if len(draws) < target:
        flags.append("low_bounded_draw_yield")
        reliable = False
    if acceptance < 0.5:
        flags.append("low_posterior_acceptance")
        reliable = False
    return draws, {
        "posterior_reliable": reliable,
        "posterior_status": "ok" if reliable else "unreliable",
        "posterior_flags": ";".join(sorted(set(flags))),
        "posterior_acceptance_fraction": acceptance,
    }


def _wise_response_pairs() -> list[tuple[str, str]]:
    pairs = []
    for band in WISE_BANDS:
        bp = bandpass_for("AllWISE", band)
        if bp and bp.svo_filter_id:
            pairs.append((str(bp.svo_filter_id), str(bp.mag_system)))
    return pairs


def _model_flux_draws_direct(
    parameter_draws: np.ndarray,
    fit: Mapping[str, object],
    library: object,
    responses: Mapping[tuple[str, str], FilterResponse],
) -> dict[str, np.ndarray]:
    output = {band: np.full(len(parameter_draws), np.nan, dtype=float) for band in WISE_BANDS}
    keys = {}
    for band in WISE_BANDS:
        bp = bandpass_for("AllWISE", band)
        if bp:
            keys[band] = (str(bp.svo_filter_id or ""), str(bp.mag_system or ""))
    logg = _safe_float(fit.get("logg")) or 4.5
    z = _safe_float(fit.get("z")) or 0.02
    rv = _safe_float(fit.get("rv")) or DEFAULT_RV
    for index, (logt, av, log_scale) in enumerate(parameter_draws):
        try:
            wavelength, spectrum = _generate_spectrum(library, float(logt), logg, z)
            observed = apply_extinction(wavelength, spectrum, float(av), rv=rv, law=EXTINCTION_LAW)
            scale = 10.0 ** float(log_scale)
            for band, key in keys.items():
                response = responses.get(key)
                if response is not None:
                    output[band][index] = scale * bandpass_flux_nu_jy(wavelength, observed, response)
        except Exception:
            continue
    return output


def _posterior_grid_nodes(values: np.ndarray, count: int) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.array([], dtype=float)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if math.isclose(lo, hi, rel_tol=0.0, abs_tol=1.0e-12):
        return np.array([lo], dtype=float)
    return np.linspace(lo, hi, max(int(count), 2), dtype=float)


def _model_flux_draws(
    parameter_draws: np.ndarray,
    fit: Mapping[str, object],
    library: object,
    responses: Mapping[tuple[str, str], FilterResponse],
    *,
    logt_nodes: int = 7,
    av_nodes: int = 5,
) -> dict[str, np.ndarray]:
    """Predict every draw from a local grid of exact bandpass integrations.

    CK04 spectrum generation is the expensive operation.  W1--W4 fluxes are
    smooth in ``(log10 Teff, Av)``, while apparent scale is exactly
    multiplicative.  We therefore evaluate CK04 and integrate the real WISE
    responses on a candidate-local grid spanning all accepted posterior draws,
    interpolate *log band flux*, and apply each draw's correlated scale.  This
    preserves one bandpass-integrated prediction per draw without thousands of
    repeated spectrum generations.
    """

    output = {band: np.full(len(parameter_draws), np.nan, dtype=float) for band in WISE_BANDS}
    if not len(parameter_draws):
        return output
    temperature_grid = _posterior_grid_nodes(parameter_draws[:, 0], logt_nodes)
    extinction_grid = _posterior_grid_nodes(parameter_draws[:, 1], av_nodes)
    if not len(temperature_grid) or not len(extinction_grid):
        return output
    keys = {}
    for band in WISE_BANDS:
        bp = bandpass_for("AllWISE", band)
        if bp:
            keys[band] = (str(bp.svo_filter_id or ""), str(bp.mag_system or ""))
    logg = _safe_float(fit.get("logg")) or 4.5
    z = _safe_float(fit.get("z")) or 0.02
    rv = _safe_float(fit.get("rv")) or DEFAULT_RV
    grids = {
        band: np.full((len(temperature_grid), len(extinction_grid)), np.nan, dtype=float)
        for band in WISE_BANDS
    }
    for temperature_index, logt in enumerate(temperature_grid):
        try:
            wavelength, spectrum = _generate_spectrum(library, float(logt), logg, z)
        except Exception:
            continue
        for extinction_index, av in enumerate(extinction_grid):
            try:
                observed = apply_extinction(
                    wavelength, spectrum, float(av), rv=rv, law=EXTINCTION_LAW
                )
            except Exception:
                continue
            for band, key in keys.items():
                response = responses.get(key)
                if response is None:
                    continue
                try:
                    value = bandpass_flux_nu_jy(wavelength, observed, response)
                except Exception:
                    continue
                if math.isfinite(value) and value > 0:
                    grids[band][temperature_index, extinction_index] = math.log10(value)
    query = parameter_draws[:, :2]
    scale = np.power(10.0, parameter_draws[:, 2])
    for band, grid in grids.items():
        if not np.all(np.isfinite(grid)):
            continue
        if len(temperature_grid) == 1 and len(extinction_grid) == 1:
            log_flux = np.full(len(query), grid[0, 0], dtype=float)
        elif len(temperature_grid) == 1:
            log_flux = np.interp(query[:, 1], extinction_grid, grid[0, :])
        elif len(extinction_grid) == 1:
            log_flux = np.interp(query[:, 0], temperature_grid, grid[:, 0])
        else:
            interpolator = RegularGridInterpolator(
                (temperature_grid, extinction_grid),
                grid,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
            log_flux = np.asarray(interpolator(query), dtype=float)
        output[band] = scale * np.power(10.0, log_flux)
    return output


def _quantiles(values: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan, np.nan, np.nan
    return tuple(float(item) for item in np.quantile(finite, [0.16, 0.50, 0.84]))


def _probability_and_z(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan, np.nan
    probability = float(np.mean(finite > 0.0))
    bounded = float(np.clip(probability, 1.0 / (2.0 * len(finite)), 1.0 - 1.0 / (2.0 * len(finite))))
    return probability, float(norm.ppf(bounded))


def _null_features(row: Mapping[str, object]) -> dict[str, float]:
    teff = _safe_float(row.get("teff_k"))
    magnitude = _safe_float(row.get("observed_mag"))
    reduced_chi2 = _safe_float(row.get("reduced_chi2"))
    av = _safe_float(row.get("av_fit"))
    gal_lat = _safe_float(row.get("galactic_latitude") or row.get("gal_b") or row.get("b"))
    return {
        "log10_teff": math.log10(teff) if teff and teff > 0 else np.nan,
        "observed_mag": magnitude if magnitude is not None else np.nan,
        "abs_galactic_latitude": abs(gal_lat) if gal_lat is not None else np.nan,
        "log10_reduced_chi2": math.log10(max(reduced_chi2, 1.0e-3)) if reduced_chi2 else np.nan,
        "av_fit": av if av is not None else np.nan,
    }


def fit_empirical_null_locus(
    control_rows: pd.DataFrame, *, minimum_controls: int = 30, version: str = "control-softl1-v1"
) -> pd.DataFrame:
    """Fit a robust conditional photospheric log-ratio locus per WISE band."""

    rows = []
    frame = pd.DataFrame() if control_rows is None else control_rows.copy()
    for band in WISE_BANDS:
        group = frame[frame.get("band", pd.Series("", index=frame.index)).astype(str).str.upper() == band].copy()
        if "quality_status" in group:
            group = group[group["quality_status"].isin(["pass", "unknown"])]
        y = pd.to_numeric(
            group.get("log_ratio_p50", pd.Series(np.nan, index=group.index, dtype=float)),
            errors="coerce",
        )
        good = pd.Series(np.isfinite(y.to_numpy(dtype=float)), index=y.index, dtype=bool)
        group = group.loc[good].copy()
        y = y.loc[good].to_numpy(dtype=float)
        base = {
            "excess_version": SED_EXCESS_VERSION, "null_locus_version": version, "band": band,
            "n_control": int(len(group)), "status": "insufficient_controls",
            "feature_names_json": "[]", "feature_centers_json": "{}", "feature_scales_json": "{}",
            "coefficients_json": "[]", "scatter_dex": np.nan, "clip_fraction": np.nan,
        }
        if len(group) < minimum_controls:
            rows.append(base)
            continue
        feature_records = [_null_features(row) for row in group.to_dict("records")]
        feature_names = [
            name for name in feature_records[0]
            if any(math.isfinite(float(record[name])) for record in feature_records if _safe_float(record[name]) is not None)
        ]
        raw = np.asarray([[record[name] for name in feature_names] for record in feature_records], dtype=float)
        centers = np.nanmedian(raw, axis=0)
        scales = np.nanpercentile(raw, 75, axis=0) - np.nanpercentile(raw, 25, axis=0)
        scales = np.where(np.isfinite(scales) & (scales > 1.0e-8), scales, 1.0)
        raw = np.where(np.isfinite(raw), raw, centers)
        x = np.column_stack([np.ones(len(raw)), (raw - centers) / scales])
        initial = np.zeros(x.shape[1], dtype=float)
        initial[0] = float(np.nanmedian(y))
        result = least_squares(lambda beta: y - x @ beta, initial, loss="soft_l1", f_scale=0.05)
        residual = y - x @ result.x
        scatter = max(1.4826 * float(np.nanmedian(np.abs(residual - np.nanmedian(residual)))), 0.01)
        keep = np.abs(residual - np.nanmedian(residual)) <= 5.0 * scatter
        if int(keep.sum()) >= minimum_controls and not np.all(keep):
            result = least_squares(lambda beta: y[keep] - x[keep] @ beta, result.x, loss="soft_l1", f_scale=scatter)
            residual = y[keep] - x[keep] @ result.x
            scatter = max(1.4826 * float(np.nanmedian(np.abs(residual - np.nanmedian(residual)))), 0.01)
        base.update({
            "status": "ok", "feature_names_json": json.dumps(feature_names, separators=(",", ":")),
            "feature_centers_json": json.dumps(dict(zip(feature_names, centers.tolist())), separators=(",", ":")),
            "feature_scales_json": json.dumps(dict(zip(feature_names, scales.tolist())), separators=(",", ":")),
            "coefficients_json": json.dumps(result.x.tolist(), separators=(",", ":")),
            "scatter_dex": scatter, "clip_fraction": float(1.0 - np.mean(keep)),
        })
        rows.append(base)
    return pd.DataFrame(rows, columns=NULL_LOCUS_COLUMNS)


def evaluate_null_locus(row: Mapping[str, object], null_locus: pd.DataFrame | None) -> tuple[float, float, str]:
    if null_locus is None or null_locus.empty:
        return 0.0, 0.0, ""
    band = str(row.get("band") or "").upper()
    match = null_locus[(null_locus["band"].astype(str).str.upper() == band) & (null_locus["status"] == "ok")]
    if match.empty:
        return 0.0, 0.0, ""
    locus = match.iloc[-1]
    names = _json_array(locus.get("feature_names_json"))
    coefficients = np.asarray(_json_array(locus.get("coefficients_json")), dtype=float)
    try:
        centers = json.loads(str(locus.get("feature_centers_json") or "{}"))
        scales = json.loads(str(locus.get("feature_scales_json") or "{}"))
    except Exception:
        return 0.0, 0.0, ""
    features = _null_features(row)
    vector = [1.0]
    for name in names:
        value = features.get(name, np.nan)
        center = float(centers.get(name, 0.0))
        scale = max(float(scales.get(name, 1.0)), 1.0e-8)
        vector.append((value - center) / scale if math.isfinite(value) else 0.0)
    if len(vector) != len(coefficients):
        return 0.0, 0.0, ""
    return float(np.dot(vector, coefficients)), float(locus.get("scatter_dex") or 0.0), str(locus.get("null_locus_version") or "")


def load_wise_variability_floors(
    external_lc_dir: str | Path | None, candidate_ids: Iterable[str] | None = None
) -> pd.DataFrame:
    """Estimate intrinsic per-band magnitude scatter from stored WISE epochs."""

    columns = ["candidate_id", "band", "variability_sigma_mag", "variability_n_points", "variability_sources"]
    if external_lc_dir is None:
        return pd.DataFrame(columns=columns)
    root = Path(external_lc_dir)
    if not root.exists():
        return pd.DataFrame(columns=columns)
    wanted = {str(value) for value in candidate_ids} if candidate_ids is not None else None
    records = []
    for prefix in ("allwise_mep_lc_", "neowise_lc_"):
        for path in root.glob(f"{prefix}*.parquet"):
            candidate_id = path.stem[len(prefix):]
            if wanted is not None and candidate_id not in wanted:
                continue
            try:
                data = pd.read_parquet(path)
            except Exception:
                continue
            for band in WISE_BANDS:
                mag_col = f"{band.lower()}mpro"
                err_col = f"{band.lower()}sigmpro"
                if mag_col not in data:
                    continue
                mag = pd.to_numeric(data[mag_col], errors="coerce")
                err = pd.to_numeric(data.get(err_col), errors="coerce") if err_col in data else pd.Series(np.nan, index=data.index)
                good = np.isfinite(mag)
                if int(good.sum()) < 3:
                    continue
                values = mag.loc[good].to_numpy(dtype=float)
                robust = 1.4826 * float(np.median(np.abs(values - np.median(values))))
                error_variance = float(np.nanmedian(np.square(err.loc[good]))) if np.isfinite(err.loc[good]).any() else 0.0
                intrinsic = math.sqrt(max(robust**2 - error_variance, 0.0))
                records.append({
                    "candidate_id": candidate_id, "band": band, "variability_sigma_mag": intrinsic,
                    "variability_n_points": int(good.sum()), "variability_sources": prefix.rstrip("_"),
                })
    if not records:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(records)
    return frame.sort_values("variability_sigma_mag").groupby(["candidate_id", "band"], as_index=False).tail(1).reset_index(drop=True)[columns]


def refresh_allwise_quality(
    candidates: pd.DataFrame,
    *,
    batch_size: int = 25,
    minimum_retry_size: int = 5,
    max_retry_depth: int = 3,
    crossmatcher: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Refresh AllWISE metadata in bounded batches, retrying failed chunks.

    CDS XMatch occasionally returns a non-VOTable error for a valid bulk
    upload.  A failed chunk is therefore bisected until it either succeeds or
    reaches ``minimum_retry_size``/``max_retry_depth``.  Successful matches are
    retained even when another chunk fails.
    """

    if candidates is None or candidates.empty:
        return pd.DataFrame() if candidates is None else candidates.copy()
    if crossmatcher is None:
        from malca.enrichment.characterize import query_allwise_vizier

        crossmatcher = query_allwise_vizier
    batch_size = max(int(batch_size), 1)
    minimum_retry_size = max(int(minimum_retry_size), 1)
    max_retry_depth = max(int(max_retry_depth), 0)
    output = candidates.copy()

    def _matched(frame: pd.DataFrame) -> pd.Series:
        if "allwise_id" not in frame:
            return pd.Series(False, index=frame.index, dtype=bool)
        values = frame["allwise_id"]
        return values.notna() & values.astype(str).str.strip().ne("") & values.astype(str).str.lower().ne("nan")

    def _refresh(indices: list[object], depth: int) -> None:
        if not indices:
            return
        refreshed = crossmatcher(output.loc[indices].copy())
        for column in refreshed.columns:
            if column not in output:
                output[column] = pd.Series(index=output.index, dtype=refreshed[column].dtype)
            output.loc[refreshed.index, column] = refreshed[column]
        unmatched = refreshed.index[~_matched(refreshed)].tolist()
        if not unmatched:
            return
        if depth >= max_retry_depth or len(unmatched) <= minimum_retry_size:
            print(f"AllWISE quality refresh: {len(unmatched)} unmatched after bounded retries")
            return
        midpoint = max(len(unmatched) // 2, 1)
        _refresh(unmatched[:midpoint], depth + 1)
        _refresh(unmatched[midpoint:], depth + 1)

    indices = output.index.tolist()
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        print(f"AllWISE quality refresh: candidates {start + 1}-{stop} of {len(indices)}")
        _refresh(indices[start:stop], 0)
    print(f"AllWISE quality refresh: {int(_matched(output).sum())}/{len(output)} matched")
    return output


def compute_best_fit_wise_ratios(
    candidates: pd.DataFrame,
    fits: pd.DataFrame,
    sed_rows: pd.DataFrame,
    *,
    library: object | None = None,
    response_loader: ResponseLoader | None = None,
    allow_bandpass_download: bool = True,
) -> pd.DataFrame:
    """Generate exact, one-evaluation WISE ratios for null-locus controls."""

    candidates = pd.DataFrame() if candidates is None else candidates.copy()
    fits = pd.DataFrame() if fits is None else fits.copy()
    sed_rows = pd.DataFrame() if sed_rows is None else sed_rows.copy()
    library = library if library is not None else _load_kurucz_library()
    responses, failures = build_response_map(
        _wise_response_pairs(), allow_download=allow_bandpass_download, response_loader=response_loader
    )
    candidate_map = {
        str(row.get("candidate_id")): row for row in candidates.to_dict("records")
        if str(row.get("candidate_id") or "")
    }
    records = []
    for fit in fits.to_dict("records"):
        candidate_id = str(fit.get("candidate_id") or "")
        candidate = candidate_map.get(candidate_id, {"candidate_id": candidate_id})
        center, _covariance, status = parse_fit_posterior(fit)
        if not np.all(np.isfinite(center)) or status == "missing_parameter_center":
            continue
        rows = sed_rows[
            (sed_rows.get("candidate_id", pd.Series("", index=sed_rows.index)).astype(str) == candidate_id)
            & sed_rows.get("source", pd.Series("", index=sed_rows.index)).astype(str).str.lower().eq("allwise")
        ].copy()
        if rows.empty:
            continue
        for idx, point in rows.iterrows():
            band = str(point.get("band") or "").upper()
            bp = bandpass_for("AllWISE", band)
            if bp and not str(point.get("svo_filter_id") or "").strip():
                rows.at[idx, "svo_filter_id"] = bp.svo_filter_id
            flags = _candidate_wise_quality_flags(candidate, band) if band in WISE_BAND_INDEX else ""
            if flags:
                rows.at[idx, "quality_flags"] = ";".join(
                    item for item in (str(point.get("quality_flags") or ""), flags) if item
                )
        prepared = _prepare_candidate_points(candidate_id, candidate, rows, responses, failures)
        predictions = _model_flux_draws(center[None, :], fit, library, responses)
        for band in WISE_BANDS:
            group = prepared[prepared["band"].astype(str).str.upper() == band].copy()
            if group.empty:
                continue
            point = group.iloc[0]
            observed = _safe_float(point.get("observed_flux_nu_jy"))
            model = _safe_float(predictions[band][0])
            if observed is None or observed <= 0 or model is None or model <= 0:
                continue
            quality = evaluate_wise_quality(point, band, candidate=candidate)
            records.append({
                **candidate,
                "candidate_id": candidate_id,
                "band": band,
                "observed_mag": _safe_float(point.get("mag")),
                "observed_flux_nu_jy": observed,
                "model_flux_nu_jy_p50": model,
                "ratio_p50": observed / model,
                "log_ratio_p50": math.log10(observed / model),
                "teff_k": _safe_float(fit.get("teff_k")),
                "av_fit": _safe_float(fit.get("av_fit")),
                "reduced_chi2": _safe_float(fit.get("reduced_chi2")),
                **quality,
            })
    return pd.DataFrame(records)


def compute_sed_excess_posteriors(
    candidates: pd.DataFrame,
    fits: pd.DataFrame,
    sed_rows: pd.DataFrame,
    *,
    sed_alpha: pd.DataFrame | None = None,
    variability: pd.DataFrame | None = None,
    null_locus: pd.DataFrame | None = None,
    n_draws: int = 2000,
    seed: int = 732451,
    library: object | None = None,
    response_loader: ResponseLoader | None = None,
    allow_bandpass_download: bool = True,
    non_simultaneity_floor_mag: float = DEFAULT_NON_SIMULTANEITY_FLOOR_MAG,
    prediction_method: str = "grid",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate model-aware WISE excess posteriors for each candidate."""

    candidates = pd.DataFrame() if candidates is None else candidates.copy()
    fits = pd.DataFrame() if fits is None else fits.copy()
    sed_rows = pd.DataFrame() if sed_rows is None else sed_rows.copy()
    library = library if library is not None else _load_kurucz_library()
    responses, failures = build_response_map(
        _wise_response_pairs(), allow_download=allow_bandpass_download, response_loader=response_loader
    )
    candidate_map = {
        str(row.get("candidate_id")): row
        for row in candidates.to_dict("records") if str(row.get("candidate_id") or "")
    }
    variability_map = {}
    if variability is not None and not variability.empty:
        variability_map = {
            (str(row["candidate_id"]), str(row["band"]).upper()): float(row["variability_sigma_mag"])
            for _, row in variability.iterrows() if _safe_float(row.get("variability_sigma_mag")) is not None
        }
    band_records: list[dict[str, object]] = []
    draw_cache: dict[str, dict[str, dict[str, np.ndarray | dict]]] = {}
    logt_bounds = _library_logt_bounds(library)
    normalized_prediction_method = str(prediction_method or "grid").strip().lower()
    if normalized_prediction_method not in {"grid", "direct"}:
        raise ValueError("prediction_method must be 'grid' or 'direct'")
    prediction_label = (
        DIRECT_MODEL_PREDICTION_METHOD
        if normalized_prediction_method == "direct"
        else MODEL_PREDICTION_METHOD
    )

    for fit in fits.to_dict("records"):
        candidate_id = str(fit.get("candidate_id") or "")
        if not candidate_id:
            continue
        candidate = candidate_map.get(candidate_id, {"candidate_id": candidate_id})
        candidate_seed = _stable_seed(candidate_id, seed)

        def append_missing_bands(
            missing_bands: Iterable[str],
            reason: str,
            posterior_meta: Mapping[str, object] | None = None,
            draw_count: int = 0,
        ) -> None:
            meta = dict(posterior_meta or {})
            meta.update({
                "posterior_reliable": False,
                "posterior_status": "not_evaluated_missing_wise",
                "posterior_flags": reason,
            })
            for missing_band in missing_bands:
                record = {
                    "candidate_id": candidate_id,
                    "excess_version": SED_EXCESS_VERSION,
                    "fit_version": str(fit.get("fit_version") or ""),
                    "fit_run_hash": str(fit.get("fit_run_hash") or ""),
                    "source": "AllWISE",
                    "band": str(missing_band),
                    "quality_pass": False,
                    "quality_status": "missing",
                    "quality_flags": "",
                    "quality_reasons": reason,
                    "n_draws": int(draw_count),
                    "random_seed": candidate_seed,
                    "model_prediction_method": prediction_label,
                    **meta,
                }
                for name in SED_EXCESS_BAND_COLUMNS:
                    record.setdefault(name, np.nan)
                band_records.append(record)

        candidate_rows = sed_rows[sed_rows.get("candidate_id", pd.Series("", index=sed_rows.index)).astype(str) == candidate_id].copy()
        wise_mask = candidate_rows.get("source", pd.Series("", index=candidate_rows.index)).astype(str).str.lower().eq("allwise")
        candidate_rows = candidate_rows.loc[wise_mask].copy()
        if candidate_rows.empty:
            append_missing_bands(WISE_BANDS, "no_wise_photometry")
            continue
        for idx, point in candidate_rows.iterrows():
            band = str(point.get("band") or "").upper()
            bp = bandpass_for("AllWISE", band)
            if bp and not str(point.get("svo_filter_id") or "").strip():
                candidate_rows.at[idx, "svo_filter_id"] = bp.svo_filter_id
            extra_flags = _candidate_wise_quality_flags(candidate, band) if band in WISE_BAND_INDEX else ""
            if extra_flags:
                candidate_rows.at[idx, "quality_flags"] = ";".join(
                    value for value in (str(point.get("quality_flags") or ""), extra_flags) if value
                )
            if _safe_float(point.get("sep_arcsec")) is None:
                candidate_rows.at[idx, "sep_arcsec"] = _safe_float(candidate.get("allwise_sep_arcsec"))
        prepared = _prepare_candidate_points(candidate_id, candidate, candidate_rows, responses, failures)
        prepared = prepared[prepared["band"].astype(str).str.upper().isin(WISE_BANDS)].copy()
        if prepared.empty:
            append_missing_bands(WISE_BANDS, "no_usable_wise_photometry")
            continue
        parameter_draws, posterior_meta = draw_fit_parameters(
            fit, n_draws, seed=candidate_seed, logt_bounds=logt_bounds
        )
        if not len(parameter_draws):
            model_draws = {band: np.array([], dtype=float) for band in WISE_BANDS}
        elif normalized_prediction_method == "direct":
            model_draws = _model_flux_draws_direct(parameter_draws, fit, library, responses)
        else:
            model_draws = _model_flux_draws(parameter_draws, fit, library, responses)
        rng = np.random.default_rng(candidate_seed ^ 0xA5A5A5A5)
        shared_mag_draw = rng.normal(0.0, max(float(non_simultaneity_floor_mag), 0.0), size=len(parameter_draws))
        draw_cache[candidate_id] = {}

        for band in WISE_BANDS:
            group = prepared[prepared["band"].astype(str).str.upper() == band].copy()
            if group.empty:
                append_missing_bands([band], "missing_band_photometry", posterior_meta, len(parameter_draws))
                continue
            group["_err"] = pd.to_numeric(group.get("observed_flux_nu_jy_err"), errors="coerce")
            point = group.sort_values("_err", na_position="last").iloc[0]
            observed = _safe_float(point.get("observed_flux_nu_jy"))
            observed_err = _safe_float(point.get("observed_flux_nu_jy_err"))
            observed_mag = _safe_float(point.get("mag"))
            observed_mag_err = _safe_float(point.get("mag_err"))
            quality = evaluate_wise_quality(point, band, candidate=candidate)
            measurement_sigma_mag = observed_mag_err
            if measurement_sigma_mag is None and observed and observed > 0 and observed_err and observed_err > 0:
                measurement_sigma_mag = 2.5 / math.log(10.0) * observed_err / observed
            measurement_sigma_mag = measurement_sigma_mag if measurement_sigma_mag is not None else np.nan
            calibration_sigma_mag = WISE_CALIBRATION_FLOOR_MAG[band]
            variability_sigma_mag = max(variability_map.get((candidate_id, band), 0.0), 0.0)
            total_sigma_mag = math.sqrt(
                (measurement_sigma_mag if math.isfinite(measurement_sigma_mag) else 0.0) ** 2
                + calibration_sigma_mag**2 + variability_sigma_mag**2
                + max(float(non_simultaneity_floor_mag), 0.0) ** 2
            )
            base = {
                "candidate_id": candidate_id, "excess_version": SED_EXCESS_VERSION,
                "fit_version": str(fit.get("fit_version") or ""), "fit_run_hash": str(fit.get("fit_run_hash") or ""),
                "source": "AllWISE", "band": band, "observed_flux_nu_jy": observed,
                "observed_flux_nu_jy_err": observed_err, "observed_mag": observed_mag,
                "observed_mag_err": observed_mag_err, "measurement_sigma_mag": measurement_sigma_mag,
                "calibration_sigma_mag": calibration_sigma_mag, "variability_sigma_mag": variability_sigma_mag,
                "non_simultaneity_sigma_mag": non_simultaneity_floor_mag,
                "total_observed_sigma_mag": total_sigma_mag, **quality, **posterior_meta,
                "n_draws": int(len(parameter_draws)), "random_seed": candidate_seed,
                "model_prediction_method": prediction_label,
            }
            for name in SED_EXCESS_BAND_COLUMNS:
                base.setdefault(name, np.nan)
            if observed is None or observed <= 0 or not len(parameter_draws):
                band_records.append(base)
                continue
            model = model_draws[band]
            valid = np.isfinite(model) & (model > 0)
            if not np.any(valid):
                base["posterior_reliable"] = False
                base["posterior_status"] = "model_prediction_failed"
                band_records.append(base)
                continue
            measurement_log_sigma = 0.4 * (measurement_sigma_mag if math.isfinite(measurement_sigma_mag) else 0.0)
            observed_log_draw = (
                math.log10(observed)
                + rng.normal(0.0, measurement_log_sigma, size=len(parameter_draws))
                + rng.normal(0.0, 0.4 * calibration_sigma_mag, size=len(parameter_draws))
                + rng.normal(0.0, 0.4 * variability_sigma_mag, size=len(parameter_draws))
                - 0.4 * shared_mag_draw
            )
            observed_draw = np.power(10.0, observed_log_draw)
            log_ratio = observed_log_draw - np.log10(model)
            ratio = np.power(10.0, log_ratio)
            model_q = _quantiles(model)
            ratio_q = _quantiles(ratio)
            excess_q = _quantiles(ratio - 1.0)
            log_q = _quantiles(log_ratio)
            p_raw, z_raw = _probability_and_z(log_ratio)
            feature_row = {**candidate, **fit, **base, "log_ratio_p50": log_q[1]}
            null_offset, null_scatter, null_version = evaluate_null_locus(feature_row, null_locus)
            calibrated_log_ratio = log_ratio - null_offset
            if null_version and null_scatter > 0:
                calibrated_log_ratio += rng.normal(0.0, null_scatter, size=len(calibrated_log_ratio))
            calibrated_ratio_q = _quantiles(np.power(10.0, calibrated_log_ratio))
            p_cal, z_cal = _probability_and_z(calibrated_log_ratio) if null_version else (np.nan, np.nan)
            base.update({
                "model_flux_nu_jy_p16": model_q[0], "model_flux_nu_jy_p50": model_q[1], "model_flux_nu_jy_p84": model_q[2],
                "ratio_p16": ratio_q[0], "ratio_p50": ratio_q[1], "ratio_p84": ratio_q[2],
                "excess_fraction_p16": excess_q[0], "excess_fraction_p50": excess_q[1], "excess_fraction_p84": excess_q[2],
                "log_ratio_p16": log_q[0], "log_ratio_p50": log_q[1], "log_ratio_p84": log_q[2],
                "p_excess": p_raw, "z_excess": z_raw, "null_offset_dex": null_offset,
                "null_scatter_dex": null_scatter, "calibrated_ratio_p16": calibrated_ratio_q[0],
                "calibrated_ratio_p50": calibrated_ratio_q[1], "calibrated_ratio_p84": calibrated_ratio_q[2],
                "p_excess_calibrated": p_cal, "z_excess_calibrated": z_cal,
                "null_locus_version": null_version,
            })
            band_records.append(base)
            draw_cache[candidate_id][band] = {
                "observed": observed_draw, "model": model, "parameter_draws": parameter_draws,
                "quality": quality, "sigma_jy": observed_err or np.nan,
            }

    bands = pd.DataFrame(band_records)
    for column in SED_EXCESS_BAND_COLUMNS:
        if column not in bands:
            bands[column] = None
    bands = bands[SED_EXCESS_BAND_COLUMNS]
    summaries = summarize_sed_excess(
        bands, sed_alpha=sed_alpha, fits=fits, draw_cache=draw_cache
    )
    return bands, summaries


def _fit_single_temperature_dust(
    candidate_id: str, cache: dict[str, dict[str, np.ndarray | dict]], fit: Mapping[str, object]
) -> dict[str, object]:
    usable = [band for band in WISE_BANDS if band in cache and bool(cache[band]["quality"].get("quality_pass"))]
    if len(usable) < 2:
        return {"dust_model": "single_temperature_blackbody", "dust_fit_status": "insufficient_clean_bands"}
    n = min(min(len(np.asarray(cache[band]["observed"])) for band in usable), 256)
    if n < 10:
        return {"dust_model": "single_temperature_blackbody", "dust_fit_status": "insufficient_draws"}
    indices = np.linspace(0, n - 1, n, dtype=int)
    wavelengths = np.asarray([bandpass_for("AllWISE", band).lambda_eff_angstrom for band in usable], dtype=float)
    frequencies = 2.99792458e18 / wavelengths
    temperature_grid = np.geomspace(50.0, 2000.0, 160)
    h = 6.62607015e-27
    k = 1.380649e-16
    c = 2.99792458e10
    sigma_sb = 5.670374419e-5
    nu = frequencies[None, :]
    temp = temperature_grid[:, None]
    bnu_jy_sr = (2.0 * h * nu**3 / c**2 / np.expm1(np.clip(h * nu / (k * temp), 1.0e-8, 700.0))) / 1.0e-23
    temperatures = []
    ratios = []
    for index in indices:
        residual = np.asarray([
            float(np.asarray(cache[band]["observed"])[index] - np.asarray(cache[band]["model"])[index])
            for band in usable
        ])
        if int((residual > 0).sum()) < 2:
            continue
        errors = np.asarray([_safe_float(cache[band].get("sigma_jy")) or max(abs(residual[pos]) * 0.1, 1.0e-9) for pos, band in enumerate(usable)])
        weight = 1.0 / np.square(np.clip(errors, 1.0e-12, None))
        numerator = np.sum(bnu_jy_sr * residual[None, :] * weight[None, :], axis=1)
        denominator = np.sum(np.square(bnu_jy_sr) * weight[None, :], axis=1)
        omega = np.clip(numerator / denominator, 0.0, None)
        chi2 = np.sum(np.square((residual[None, :] - omega[:, None] * bnu_jy_sr) / errors[None, :]), axis=1)
        best = int(np.nanargmin(chi2))
        apparent_scale = 10.0 ** float(np.asarray(cache[usable[0]]["parameter_draws"])[index, 2])
        star_bolometric_flux = apparent_scale * LSUN_ERG_S
        dust_bolometric_flux = omega[best] * sigma_sb * temperature_grid[best] ** 4 / math.pi
        if star_bolometric_flux > 0 and dust_bolometric_flux >= 0:
            temperatures.append(temperature_grid[best])
            ratios.append(dust_bolometric_flux / star_bolometric_flux)
    if len(ratios) < 10:
        return {"dust_model": "single_temperature_blackbody", "dust_fit_status": "insufficient_positive_residual_draws"}
    tq = _quantiles(np.asarray(temperatures))
    rq = _quantiles(np.asarray(ratios))
    return {
        "dust_model": "single_temperature_blackbody", "dust_fit_status": "ok",
        "dust_temperature_k_p16": tq[0], "dust_temperature_k_p50": tq[1], "dust_temperature_k_p84": tq[2],
        "lir_lstar_p16": rq[0], "lir_lstar_p50": rq[1], "lir_lstar_p84": rq[2],
    }


def summarize_sed_excess(
    bands: pd.DataFrame,
    *,
    sed_alpha: pd.DataFrame | None = None,
    fits: pd.DataFrame | None = None,
    draw_cache: dict[str, dict[str, dict[str, np.ndarray | dict]]] | None = None,
) -> pd.DataFrame:
    records = []
    alpha_map = {}
    if sed_alpha is not None and not sed_alpha.empty:
        alpha_map = {str(row["candidate_id"]): row for _, row in sed_alpha.iterrows()}
    fit_map = {}
    if fits is not None and not fits.empty:
        fit_map = {str(row["candidate_id"]): row for _, row in fits.iterrows()}
    for candidate_id, group in bands.groupby("candidate_id", sort=True):
        by_band = {str(row["band"]).upper(): row for _, row in group.iterrows()}
        assessable = {
            band: row for band, row in by_band.items()
            if bool(row.get("quality_pass")) and bool(row.get("posterior_reliable"))
            and _safe_float(row.get("p_excess_calibrated")) is not None
        }
        strong = {
            band for band, row in assessable.items()
            if band in {"W3", "W4"} and float(row["p_excess_calibrated"]) >= PRIMARY_EXCESS_PROBABILITY
        }
        support = {
            band for band, row in assessable.items()
            if float(row["p_excess_calibrated"]) >= ADJACENT_SUPPORT_PROBABILITY
        }
        classification = "none"
        reason = "clean_bands_consistent_with_photosphere"
        primary = ""
        adjacent = ""
        if not assessable:
            missing_reasons = {
                str(value) for value in group.get("quality_reasons", pd.Series(dtype=object)).dropna()
                if str(value).startswith("no_") or str(value) == "missing_band_photometry"
            }
            if missing_reasons:
                classification = "unassessable"
                reason = sorted(missing_reasons)[0]
            elif not group["posterior_reliable"].fillna(False).astype(bool).any():
                classification, reason = "unassessable", "posterior_unreliable"
            elif group["p_excess_calibrated"].notna().any():
                classification, reason = "unassessable", "no_quality_pass_reliable_band"
            elif not group["null_locus_version"].fillna("").astype(str).str.strip().ne("").any():
                classification, reason = "uncalibrated", "no_empirical_null_locus"
            else:
                classification, reason = "unassessable", "no_calibrated_posterior"
        elif "W3" in strong and ({"W2", "W4"} & support):
            classification, reason, primary = "robust", "w3_with_adjacent_support", "W3"
            adjacent = sorted({"W2", "W4"} & support)[0]
        elif "W4" in strong and "W3" in support:
            classification, reason, primary, adjacent = "robust", "w4_with_w3_support", "W4", "W3"
        elif "W4" in strong and "W3" in assessable:
            classification, reason, primary = "isolated_w4", "w4_without_w3_support", "W4"
        elif strong:
            primary = sorted(strong)[0]
            classification, reason = "single_band_candidate", "strong_primary_without_assessable_adjacent_band"
        else:
            probable = [
                band for band in ("W3", "W4") if band in assessable
                and float(assessable[band]["p_excess_calibrated"]) >= PROBABLE_EXCESS_PROBABILITY
            ]
            if probable:
                classification, reason, primary = "probable", "moderate_calibrated_excess_probability", probable[0]
        alpha = alpha_map.get(str(candidate_id), {})
        record = {
            "candidate_id": str(candidate_id), "excess_version": SED_EXCESS_VERSION,
            "excess_class": classification, "classification_reason": reason,
            "primary_band": primary, "adjacent_support": adjacent,
            "posterior_reliable": bool(group["posterior_reliable"].all()),
            "quality_summary": ";".join(f"{band}:{row.get('quality_status')}" for band, row in sorted(by_band.items())),
            "sed_alpha": alpha.get("sed_alpha", np.nan), "sed_alpha_class": alpha.get("sed_alpha_class", "unknown"),
        }
        for band in ("W3", "W4"):
            row = by_band.get(band, {})
            prefix = band.lower()
            record[f"{prefix}_ratio_p50"] = row.get("ratio_p50", np.nan)
            record[f"{prefix}_ratio_p16"] = row.get("ratio_p16", np.nan)
            record[f"{prefix}_ratio_p84"] = row.get("ratio_p84", np.nan)
            record[f"{prefix}_p_excess_calibrated"] = row.get("p_excess_calibrated", np.nan)
        if classification in {"robust", "probable"} and draw_cache and str(candidate_id) in draw_cache:
            record.update(_fit_single_temperature_dust(str(candidate_id), draw_cache[str(candidate_id)], fit_map.get(str(candidate_id), {})))
        else:
            record.update({
                "dust_model": "single_temperature_blackbody",
                "dust_fit_status": "not_run_noncoherent_excess" if classification not in {"robust", "probable"} else "not_run",
            })
        records.append(record)
    out = pd.DataFrame(records)
    for column in SED_EXCESS_SUMMARY_COLUMNS:
        if column not in out:
            out[column] = None
    return out[SED_EXCESS_SUMMARY_COLUMNS]


def ensure_sed_excess_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS sed_excess_bands (
        candidate_id TEXT NOT NULL, excess_version TEXT NOT NULL, fit_version TEXT, fit_run_hash TEXT,
        source TEXT, band TEXT NOT NULL, payload_json TEXT NOT NULL,
        PRIMARY KEY(candidate_id, excess_version, band))"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS sed_excess_summary (
        candidate_id TEXT NOT NULL, excess_version TEXT NOT NULL, excess_class TEXT,
        payload_json TEXT NOT NULL, PRIMARY KEY(candidate_id, excess_version))"""
    )


def upsert_sed_excess_results(conn: sqlite3.Connection, bands: pd.DataFrame, summaries: pd.DataFrame) -> tuple[int, int]:
    ensure_sed_excess_schema(conn)
    n_band = 0
    for record in bands.to_dict("records"):
        payload = json.dumps(record, default=_json_default, separators=(",", ":"))
        conn.execute(
            "INSERT INTO sed_excess_bands VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(candidate_id, excess_version, band) DO UPDATE SET fit_version=excluded.fit_version, fit_run_hash=excluded.fit_run_hash, source=excluded.source, payload_json=excluded.payload_json",
            (str(record["candidate_id"]), str(record["excess_version"]), str(record.get("fit_version") or ""), str(record.get("fit_run_hash") or ""), str(record.get("source") or ""), str(record["band"]), payload),
        )
        n_band += 1
    n_summary = 0
    for record in summaries.to_dict("records"):
        payload = json.dumps(record, default=_json_default, separators=(",", ":"))
        conn.execute(
            "INSERT INTO sed_excess_summary VALUES (?, ?, ?, ?) ON CONFLICT(candidate_id, excess_version) DO UPDATE SET excess_class=excluded.excess_class, payload_json=excluded.payload_json",
            (str(record["candidate_id"]), str(record["excess_version"]), str(record.get("excess_class") or ""), payload),
        )
        n_summary += 1
    conn.commit()
    return n_band, n_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="malca sed-excess", description="Compute model-aware WISE SED excess posteriors")
    parser.add_argument("review_db", type=Path)
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--event-class", default="dipper")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--null-locus", type=Path, default=None)
    parser.add_argument("--control-sample-size", type=int, default=2000)
    parser.add_argument("--skip-null-calibration", action="store_true")
    parser.add_argument("--external-lc-dir", type=Path, default=None)
    parser.add_argument(
        "--refresh-wise-quality",
        action="store_true",
        help=(
            "Targeted, batched AllWISE XMatch for selected candidates so legacy review "
            "DBs gain band quality metadata in the generated sidecars."
        ),
    )
    parser.add_argument(
        "--refresh-control-wise-quality",
        action="store_true",
        help="Also refresh AllWISE quality metadata for null controls (slow and normally unnecessary).",
    )
    parser.add_argument(
        "--wise-quality-batch-size",
        type=int,
        default=25,
        help="Number of sources per AllWISE XMatch upload (default: 25).",
    )
    parser.add_argument("--n-draws", type=int, default=2000)
    parser.add_argument(
        "--prediction-method",
        choices=("grid", "direct"),
        default="grid",
        help=(
            "Use the validated local bandpass-response grid (default) or exact CK04 "
            "spectrum generation for every posterior draw (much slower)."
        ),
    )
    parser.add_argument("--seed", type=int, default=732451)
    parser.add_argument("--write-review-db", action="store_true")
    parser.add_argument("--no-bandpass-download", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    review_db = Path(args.review_db)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(review_db) as conn:
        candidates = pd.read_sql_query("SELECT * FROM candidates", conn)
        requested = {str(item) for item in (getattr(args, "candidate_id", []) or [])}
        if requested:
            candidates = candidates[candidates["candidate_id"].astype(str).isin(requested)].copy()
        elif str(getattr(args, "event_class", "") or ""):
            reviewed = pd.read_sql_query(
                "SELECT candidate_id FROM reviews WHERE event_class = ?", conn,
                params=(str(args.event_class),),
            )
            candidates = candidates[candidates["candidate_id"].astype(str).isin(reviewed["candidate_id"].astype(str))].copy()
        ids = candidates["candidate_id"].astype(str).tolist()
        if not ids:
            raise ValueError("No candidates selected")
        placeholders = ",".join(["?"] * len(ids))
        fits = pd.read_sql_query(f"SELECT * FROM sed_model_fits WHERE candidate_id IN ({placeholders})", conn, params=ids)
        sed_rows = pd.read_sql_query(f"SELECT * FROM sed_photometry WHERE candidate_id IN ({placeholders})", conn, params=ids)
        alpha_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(candidates)")}
        alpha = candidates[[column for column in ("candidate_id", "sed_alpha", "sed_alpha_class") if column in alpha_columns]].copy()
        control_candidates = pd.DataFrame()
        control_fits = pd.DataFrame()
        control_sed_rows = pd.DataFrame()
        need_controls = not bool(getattr(args, "skip_null_calibration", False)) and not getattr(args, "null_locus", None)
        if need_controls:
            requested_controls = max(int(getattr(args, "control_sample_size", 2000)), 30)
            control_fits = pd.read_sql_query(
                "SELECT * FROM sed_model_fits WHERE status = 'ok' AND fit_version = ? "
                "ORDER BY candidate_id LIMIT ?",
                conn, params=(SED_MODEL_FIT_VERSION, requested_controls + len(ids)),
            )
            control_fits = control_fits[~control_fits["candidate_id"].astype(str).isin(ids)].head(requested_controls).copy()
            control_ids = control_fits["candidate_id"].astype(str).tolist()
            if control_ids:
                control_placeholders = ",".join(["?"] * len(control_ids))
                control_candidates = pd.read_sql_query(
                    f"SELECT * FROM candidates WHERE candidate_id IN ({control_placeholders})", conn, params=control_ids
                )
                control_sed_rows = pd.read_sql_query(
                    f"SELECT * FROM sed_photometry WHERE candidate_id IN ({control_placeholders})", conn, params=control_ids
                )
    quality_path = output_dir / "marked_dipper_allwise_quality.parquet"
    if bool(getattr(args, "refresh_wise_quality", False)):
        candidates = refresh_allwise_quality(
            candidates,
            batch_size=int(getattr(args, "wise_quality_batch_size", 25)),
        )
        quality_columns = [
            column for column in candidates.columns
            if column == "candidate_id" or column.startswith("allwise_")
        ]
        candidates[quality_columns].to_parquet(quality_path, index=False)
    elif quality_path.exists():
        cached_quality = pd.read_parquet(quality_path)
        cached_columns = [column for column in cached_quality.columns if column != "candidate_id"]
        candidates = candidates.drop(columns=[column for column in cached_columns if column in candidates], errors="ignore")
        candidates = candidates.merge(cached_quality, on="candidate_id", how="left", validate="one_to_one")
    if bool(getattr(args, "refresh_control_wise_quality", False)) and not control_candidates.empty:
        control_candidates = refresh_allwise_quality(
            control_candidates,
            batch_size=int(getattr(args, "wise_quality_batch_size", 25)),
        )
    missing_covariance = fits.get("fit_covariance_json", pd.Series("", index=fits.index)).fillna("").astype(str).isin(["", "[]"])
    if not (fits.get("fit_version", pd.Series("", index=fits.index)) == SED_MODEL_FIT_VERSION).all():
        raise RuntimeError(
            f"Selected fit(s) are not {SED_MODEL_FIT_VERSION}; "
            "run `malca sed-fit REVIEW_DB --event-class dipper` or targeted --candidate-id backfills first"
        )
    if missing_covariance.any():
        print(
            f"{int(missing_covariance.sum())} selected fit(s) lack covariance; "
            "their WISE excess entries will be emitted as posterior-unreliable."
        )
    null_path = getattr(args, "null_locus", None)
    if null_path:
        null_locus = pd.read_csv(null_path) if str(null_path).lower().endswith(".csv") else pd.read_parquet(null_path)
    elif bool(getattr(args, "skip_null_calibration", False)):
        null_locus = None
    else:
        control_ratios = compute_best_fit_wise_ratios(
            control_candidates, control_fits, control_sed_rows,
            allow_bandpass_download=not bool(getattr(args, "no_bandpass_download", False)),
        )
        control_path = output_dir / "sed_excess_null_controls.parquet"
        control_ratios.to_parquet(control_path, index=False)
        null_locus = fit_empirical_null_locus(control_ratios)
        null_path = output_dir / "sed_excess_null_locus.parquet"
        null_locus.to_parquet(null_path, index=False)
    variability = load_wise_variability_floors(getattr(args, "external_lc_dir", None), ids)
    bands, summaries = compute_sed_excess_posteriors(
        candidates, fits, sed_rows, sed_alpha=alpha, variability=variability, null_locus=null_locus,
        n_draws=max(int(args.n_draws), 1), seed=int(args.seed),
        allow_bandpass_download=not bool(getattr(args, "no_bandpass_download", False)),
        prediction_method=str(getattr(args, "prediction_method", "grid")),
    )
    band_path = output_dir / "marked_dipper_sed_excess_bands.parquet"
    summary_path = output_dir / "marked_dipper_sed_excess_summary.csv"
    bands.to_parquet(band_path, index=False)
    summaries.to_csv(summary_path, index=False)
    if bool(getattr(args, "write_review_db", False)):
        with sqlite3.connect(review_db) as conn:
            upsert_sed_excess_results(conn, bands, summaries)
    return {
        "candidates": len(summaries), "band_rows": len(bands), "bands": band_path,
        "summary": summary_path, "null_locus": null_path,
        "allwise_quality": quality_path if quality_path.exists() else None,
    }


def main(argv: list[str] | None = None) -> None:
    result = run(build_arg_parser().parse_args(argv))
    print(f"Wrote {result['band_rows']} band posterior rows for {result['candidates']} candidates")
    print(result["bands"])
    print(result["summary"])
    if result.get("null_locus"):
        print(result["null_locus"])


if __name__ == "__main__":
    main()
