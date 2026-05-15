"""Castelli/Kurucz stellar-atmosphere fitting for broadband SED rows."""

from __future__ import annotations

import importlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


SED_MODEL_FIT_TABLE_NAME = "sed_model_fits"
SED_MODEL_CURVE_TABLE_NAME = "sed_model_curves"

SED_MODEL_FIT_COLUMNS = [
    "candidate_id",
    "model_family",
    "teff_k",
    "logg",
    "z",
    "av_fixed",
    "scale",
    "luminosity_lsun",
    "radius_rsun",
    "chi2",
    "reduced_chi2",
    "n_fit_points",
    "fit_lambda_min",
    "fit_lambda_max",
    "fit_bands_json",
    "status",
    "warning",
]

SED_MODEL_CURVE_COLUMNS = [
    "candidate_id",
    "model_family",
    "wavelength_angstrom",
    "lambda_l_lambda",
    "flux_lambda",
    "teff_k",
    "scale",
]

MODEL_FAMILY = "Castelli/Kurucz 2004"
LSUN_ERG_S = 3.828e33
PC_CM = 3.085677581491367e18
MIN_FIT_POINTS = 3
DEFAULT_SIGMA_LOG = 0.08
MIN_SIGMA_LOG = 0.02
DEFAULT_LOGT_BOUNDS = (3.54406, 4.699)
FIT_LAMBDA_MIN_ANGSTROM = 3000.0
FIT_LAMBDA_MAX_ANGSTROM = 10000.0

_EXCLUDED_SOURCE_TOKENS = (
    "galex",
    "2mass",
    "allwise",
    "wise",
    "spitzer",
    "akari",
    "iras",
    "herschel",
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


def _empty_fit_row(candidate_id: str, *, status: str, warning: str, row: pd.Series | dict | None = None) -> dict:
    row = row if row is not None else {}
    logg = _logg_from_candidate(row)
    z = _z_from_candidate(row)
    av = _extinction_av_from_candidate(row)
    return {
        "candidate_id": str(candidate_id),
        "model_family": MODEL_FAMILY,
        "teff_k": np.nan,
        "logg": logg,
        "z": z,
        "av_fixed": av,
        "scale": np.nan,
        "luminosity_lsun": np.nan,
        "radius_rsun": np.nan,
        "chi2": np.nan,
        "reduced_chi2": np.nan,
        "n_fit_points": 0,
        "fit_lambda_min": np.nan,
        "fit_lambda_max": np.nan,
        "fit_bands_json": "[]",
        "status": status,
        "warning": warning,
    }


def _is_stellar_fit_band(row: pd.Series) -> bool:
    source = str(row.get("source") or "").strip().lower()
    band = str(row.get("band") or "").strip().lower()
    flags = str(row.get("quality_flags") or "").strip().lower()
    lam = _safe_float(row.get("lambda_eff_angstrom"))
    if lam is None or lam < FIT_LAMBDA_MIN_ANGSTROM or lam > FIT_LAMBDA_MAX_ANGSTROM:
        return False
    if any(token in source for token in _EXCLUDED_SOURCE_TOKENS):
        return False
    if "halpha" in band or band in {"ha", "h-alpha", "h_alpha"}:
        return False
    if "confusion_risk" in flags:
        return False
    if _to_bool(row.get("is_upper_limit", False)):
        return False
    return True


def _prepare_fit_points(candidate_id: str, candidate: pd.Series | dict, sed_rows: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if sed_rows is None or sed_rows.empty or "candidate_id" not in sed_rows.columns:
        return pd.DataFrame(), "lambda_l_lambda"

    frame = sed_rows[sed_rows["candidate_id"].astype(str) == str(candidate_id)].copy()
    if frame.empty:
        return pd.DataFrame(), "lambda_l_lambda"
    already_corrected = False
    if "sed_mode" in frame.columns:
        sed_mode = frame["sed_mode"].fillna("").astype(str).str.strip().str.lower()
        corrected_mask = sed_mode.isin({"ism-corrected", "ism_corrected", "corrected", "dereddened"})
        if corrected_mask.any():
            frame = frame.loc[corrected_mask].copy()
            already_corrected = True
    frame = frame[frame.apply(_is_stellar_fit_band, axis=1)].copy()
    if frame.empty:
        return frame, "lambda_l_lambda"

    lum = pd.to_numeric(frame.get("lambda_l_lambda"), errors="coerce")
    flux = pd.to_numeric(frame.get("flux_lambda"), errors="coerce")
    y_col = "lambda_l_lambda" if np.isfinite(lum).sum() >= MIN_FIT_POINTS else "flux_lambda"
    err_col = "lambda_l_lambda_err" if y_col == "lambda_l_lambda" else "flux_lambda_err"
    y = pd.to_numeric(frame.get(y_col), errors="coerce").astype(float)
    yerr = pd.to_numeric(frame.get(err_col), errors="coerce").astype(float) if err_col in frame else pd.Series(np.nan, index=frame.index)
    lam = pd.to_numeric(frame.get("lambda_eff_angstrom"), errors="coerce").astype(float)

    av = _extinction_av_from_candidate(candidate)
    av_coeff = pd.to_numeric(frame.get("av_coeff"), errors="coerce").astype(float) if "av_coeff" in frame else pd.Series(np.nan, index=frame.index)
    correction = np.ones(len(frame), dtype=float)
    if not already_corrected:
        good_coeff = np.isfinite(av_coeff.to_numpy(dtype=float))
        correction[good_coeff] = 10.0 ** (0.4 * av * av_coeff.to_numpy(dtype=float)[good_coeff])

    out = frame.copy()
    out["fit_lambda"] = lam
    out["fit_y"] = y.to_numpy(dtype=float) * correction
    out["fit_yerr"] = yerr.to_numpy(dtype=float) * correction
    good = np.isfinite(out["fit_lambda"]) & np.isfinite(out["fit_y"]) & (out["fit_lambda"] > 0) & (out["fit_y"] > 0)
    out = out.loc[good].copy()
    if out.empty:
        return out, y_col

    rel_err = np.abs(pd.to_numeric(out["fit_yerr"], errors="coerce").to_numpy(dtype=float) / out["fit_y"].to_numpy(dtype=float))
    sigma_log = rel_err / math.log(10.0)
    sigma_log[~np.isfinite(sigma_log) | (sigma_log <= 0)] = DEFAULT_SIGMA_LOG
    sigma_log = np.clip(sigma_log, MIN_SIGMA_LOG, 0.5)
    out["fit_sigma_log"] = sigma_log
    return out, y_col


def _fit_single_candidate(
    candidate_id: str,
    candidate: pd.Series | dict,
    sed_rows: pd.DataFrame,
    library: object,
    *,
    curve_points: int = 400,
) -> tuple[dict, pd.DataFrame]:
    points, y_col = _prepare_fit_points(candidate_id, candidate, sed_rows)
    if len(points) < MIN_FIT_POINTS:
        fit_row = _empty_fit_row(
            candidate_id,
            status="insufficient_data",
            warning=f"Need at least {MIN_FIT_POINTS} finite optical photospheric SED points; found {len(points)}.",
            row=candidate,
        )
        return fit_row, pd.DataFrame(columns=SED_MODEL_CURVE_COLUMNS)

    logg = _logg_from_candidate(candidate)
    z = _z_from_candidate(candidate)
    av = _extinction_av_from_candidate(candidate)
    distance_pc = _distance_pc_from_candidate(candidate)
    bounds = _library_logt_bounds(library)
    teff_initial = _teff_initial_from_candidate(candidate, bounds)
    logt_initial = math.log10(teff_initial)
    lo, hi = bounds
    lo = max(float(lo), 3.2)
    hi = min(float(hi), 5.0)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        lo, hi = DEFAULT_LOGT_BOUNDS

    x = points["fit_lambda"].to_numpy(dtype=float)
    obs_log = np.log10(points["fit_y"].to_numpy(dtype=float))
    sigma_log = points["fit_sigma_log"].to_numpy(dtype=float)
    weights = 1.0 / np.square(sigma_log)
    spectrum_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def spectrum_for(logt: float) -> tuple[np.ndarray, np.ndarray]:
        key = round(float(logt), 6)
        if key not in spectrum_cache:
            spectrum_cache[key] = _generate_spectrum(library, float(logt), logg, z)
        return spectrum_cache[key]

    def evaluate(logt: float) -> tuple[float, float]:
        try:
            wavelength, spectrum = spectrum_for(logt)
            if y_col == "lambda_l_lambda":
                model = x * np.interp(x, wavelength, spectrum, left=np.nan, right=np.nan)
            else:
                model = np.interp(x, wavelength, spectrum, left=np.nan, right=np.nan)
        except Exception:
            return 1.0e99, np.nan
        if not np.all(np.isfinite(model) & (model > 0)):
            return 1.0e99, np.nan
        model_log = np.log10(model)
        log_scale = float(np.sum(weights * (obs_log - model_log)) / np.sum(weights))
        resid = (obs_log - model_log - log_scale) / sigma_log
        return float(np.sum(np.square(resid))), log_scale

    grid = np.unique(np.concatenate([np.linspace(lo, hi, 96), np.array([np.clip(logt_initial, lo, hi)])]))
    grid_values = np.array([evaluate(logt)[0] for logt in grid], dtype=float)
    finite_grid = np.isfinite(grid_values) & (grid_values < 1.0e98)
    if not np.any(finite_grid):
        fit_row = _empty_fit_row(
            candidate_id,
            status="fit_failed",
            warning="No valid Kurucz spectra were generated within the fitted Teff range.",
            row=candidate,
        )
        return fit_row, pd.DataFrame(columns=SED_MODEL_CURVE_COLUMNS)

    finite_indices = np.where(finite_grid)[0]
    best_grid_index = int(finite_indices[np.argmin(grid_values[finite_grid])])
    left_index = max(0, best_grid_index - 2)
    right_index = min(len(grid) - 1, best_grid_index + 2)
    opt_lo = float(grid[left_index])
    opt_hi = float(grid[right_index])
    if opt_lo >= opt_hi:
        opt_lo, opt_hi = lo, hi

    result = minimize_scalar(lambda logt: evaluate(float(logt))[0], bounds=(opt_lo, opt_hi), method="bounded")
    if not result.success or not math.isfinite(float(result.fun)):
        logt_best = float(grid[best_grid_index])
        chi2, log_scale = evaluate(logt_best)
    else:
        logt_best = float(result.x)
        chi2, log_scale = evaluate(logt_best)

    if not (math.isfinite(chi2) and chi2 < 1.0e98 and math.isfinite(log_scale)):
        fit_row = _empty_fit_row(candidate_id, status="fit_failed", warning="Kurucz fit optimizer did not converge.", row=candidate)
        return fit_row, pd.DataFrame(columns=SED_MODEL_CURVE_COLUMNS)

    scale = float(10.0 ** log_scale)
    teff = float(10.0 ** logt_best)
    dof = max(int(len(points)) - 2, 1)
    reduced_chi2 = float(chi2 / dof)
    luminosity_lsun = scale if y_col == "lambda_l_lambda" else np.nan
    radius_rsun = (
        math.sqrt(luminosity_lsun) * (5772.0 / teff) ** 2
        if math.isfinite(luminosity_lsun) and luminosity_lsun > 0 and teff > 0
        else np.nan
    )
    band_records = [
        {
            "source": str(row.get("source") or ""),
            "band": str(row.get("band") or ""),
            "lambda_eff_angstrom": _safe_float(row.get("lambda_eff_angstrom")),
        }
        for _, row in points.iterrows()
    ]
    fit_row = {
        "candidate_id": str(candidate_id),
        "model_family": MODEL_FAMILY,
        "teff_k": teff,
        "logg": logg,
        "z": z,
        "av_fixed": av,
        "scale": scale,
        "luminosity_lsun": luminosity_lsun,
        "radius_rsun": radius_rsun,
        "chi2": float(chi2),
        "reduced_chi2": reduced_chi2,
        "n_fit_points": int(len(points)),
        "fit_lambda_min": float(np.nanmin(points["fit_lambda"])),
        "fit_lambda_max": float(np.nanmax(points["fit_lambda"])),
        "fit_bands_json": json.dumps(band_records, separators=(",", ":")),
        "status": "ok",
        "warning": "",
    }

    wavelength, spectrum = spectrum_for(logt_best)
    curve_min = max(float(np.nanmin(wavelength)), 1000.0)
    curve_max = min(float(np.nanmax(wavelength)), 250000.0)
    if curve_min >= curve_max:
        curve_wave = wavelength
    else:
        curve_wave = np.geomspace(curve_min, curve_max, max(int(curve_points), 32))
    curve_spectrum = np.interp(curve_wave, wavelength, spectrum, left=np.nan, right=np.nan)
    curve_l_lambda = scale * curve_wave * curve_spectrum
    if distance_pc is not None and distance_pc > 0:
        distance_cm = distance_pc * PC_CM
        curve_flux = scale * curve_spectrum / (4.0 * math.pi * distance_cm * distance_cm)
    else:
        curve_flux = np.full_like(curve_l_lambda, np.nan, dtype=float)
    curve = pd.DataFrame({
        "candidate_id": str(candidate_id),
        "model_family": MODEL_FAMILY,
        "wavelength_angstrom": curve_wave,
        "lambda_l_lambda": curve_l_lambda,
        "flux_lambda": curve_flux,
        "teff_k": teff,
        "scale": scale,
    })
    curve = curve[np.isfinite(curve["wavelength_angstrom"]) & np.isfinite(curve["lambda_l_lambda"]) & (curve["lambda_l_lambda"] > 0)]
    return fit_row, curve[SED_MODEL_CURVE_COLUMNS].reset_index(drop=True)


def fit_sed_models(
    candidates: pd.DataFrame,
    sed_rows: pd.DataFrame,
    *,
    library: object | None = None,
    curve_points: int = 400,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit Castelli/Kurucz atmosphere models to normalized SED photometry rows."""
    library = library if library is not None else _load_kurucz_library()
    candidates = pd.DataFrame() if candidates is None else candidates.copy()
    sed_rows = pd.DataFrame() if sed_rows is None else sed_rows.copy()

    candidate_map: dict[str, pd.Series | dict] = {}
    if not candidates.empty:
        for idx, row in candidates.iterrows():
            cid = _candidate_id_for_row(row, idx)
            candidate_map[str(cid)] = row
    if not sed_rows.empty and "candidate_id" in sed_rows.columns:
        for cid in sed_rows["candidate_id"].dropna().astype(str).unique():
            candidate_map.setdefault(str(cid), {"candidate_id": str(cid)})

    fit_rows: list[dict] = []
    curve_parts: list[pd.DataFrame] = []
    total = len(candidate_map)
    for idx, (candidate_id, candidate) in enumerate(candidate_map.items(), start=1):
        if progress_callback and (idx == 1 or idx % 50 == 0 or idx == total):
            progress_callback(f"[SED model] fitting {idx}/{total}: {candidate_id}")
        try:
            fit_row, curve = _fit_single_candidate(
                candidate_id,
                candidate,
                sed_rows,
                library,
                curve_points=curve_points,
            )
        except PystellibsSetupError:
            raise
        except Exception as exc:
            fit_row = _empty_fit_row(candidate_id, status="fit_failed", warning=str(exc), row=candidate)
            curve = pd.DataFrame(columns=SED_MODEL_CURVE_COLUMNS)
        fit_rows.append(fit_row)
        if isinstance(curve, pd.DataFrame) and not curve.empty:
            curve_parts.append(curve)

    fits = pd.DataFrame(fit_rows, columns=SED_MODEL_FIT_COLUMNS)
    if curve_parts:
        curves = pd.concat(curve_parts, ignore_index=True)
    else:
        curves = pd.DataFrame(columns=SED_MODEL_CURVE_COLUMNS)
    for col in SED_MODEL_FIT_COLUMNS:
        if col not in fits.columns:
            fits[col] = None
    for col in SED_MODEL_CURVE_COLUMNS:
        if col not in curves.columns:
            curves[col] = None
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


def upsert_sed_model_fits(conn: sqlite3.Connection, fits: pd.DataFrame) -> int:
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
    conn.commit()
    return count


def upsert_sed_model_curves(
    conn: sqlite3.Connection,
    curves: pd.DataFrame,
    *,
    replace_candidate_ids: Iterable[str] | None = None,
) -> int:
    candidate_ids = [str(cid) for cid in (replace_candidate_ids or [])]
    if curves is not None and not curves.empty and "candidate_id" in curves.columns:
        candidate_ids.extend(str(cid) for cid in curves["candidate_id"].dropna().astype(str).unique())
    for cid in sorted(set(candidate_ids)):
        conn.execute(f"DELETE FROM {SED_MODEL_CURVE_TABLE_NAME} WHERE candidate_id = ?", (cid,))

    if curves is None or curves.empty:
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
    conn.commit()
    return count


def upsert_sed_model_results(
    conn: sqlite3.Connection,
    fits: pd.DataFrame,
    curves: pd.DataFrame,
    *,
    replace_candidate_ids: Iterable[str] | None = None,
) -> tuple[int, int]:
    n_fits = upsert_sed_model_fits(conn, fits)
    candidate_ids = replace_candidate_ids
    if candidate_ids is None and fits is not None and not fits.empty and "candidate_id" in fits.columns:
        candidate_ids = fits["candidate_id"].dropna().astype(str).unique().tolist()
    n_curves = upsert_sed_model_curves(conn, curves, replace_candidate_ids=candidate_ids)
    return n_fits, n_curves
