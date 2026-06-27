from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from malca.config import BAD_CAMERA_SCATTER_RATIO_THRESHOLD
from malca.core.utils import filter_bad_cameras, read_lc_dat2


CAMERA_COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363",
]

ASASSN_COLUMNS = [
    "JD",
    "mag",
    "error",
    "good_bad",
    "camera#",
    "v_g_band",
    "saturated",
    "cam_field",
]

CANONICAL_ASASSN_COLUMNS = [
    "jd",
    "mjd",
    "band",
    "mag",
    "mag_err",
    "flux",
    "flux_err",
    "flux_density_mjy",
    "flux_density_mjy_err",
    "rel_flux",
    "rel_flux_err",
    "flux_provenance",
    "quality",
    "camera",
    "limit",
    "fwhm",
    "source_path",
    "is_good",
    "saturated",
    "camera_name",
    "field",
]

MJD_OFFSET = 2400000.5
ASASSN_REDUCED_JD_OFFSET = 2450000.0
AB_ZERO_POINT_JY = 3631.0
V_VEGA_ZERO_POINT_JY = 3636.0
MAG_ERROR_TO_FRAC_FLUX = np.log(10.0) * 0.4

_GOOD_STRINGS = {"1", "G", "GOOD", "T", "TRUE", "Y", "YES", "OK"}
_BAD_STRINGS = {"0", "B", "BAD", "F", "FALSE", "N", "NO"}


def stable_camera_color(camera_label: str) -> str:
    """Return a deterministic color for a camera label across plots."""
    s = str(camera_label)
    try:
        idx = int(s) % len(CAMERA_COLOR_PALETTE)
    except Exception:
        digest = hashlib.md5(s.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % len(CAMERA_COLOR_PALETTE)
    return CAMERA_COLOR_PALETTE[idx]


def read_asassn_dat(dat_path: str | Path) -> pd.DataFrame:
    """Read an ASAS-SN `.dat` light curve using whitespace separation."""
    return pd.read_csv(
        dat_path,
        sep=r"\s+",
        names=ASASSN_COLUMNS,
        dtype={
            "JD": float,
            "mag": float,
            "error": float,
            "good_bad": int,
            "camera#": int,
            "v_g_band": int,
            "saturated": int,
            "cam_field": str,
        },
        comment="#",
    )


def _empty_canonical() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_ASASSN_COLUMNS)


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower = {str(col).strip().lower(): col for col in df.columns}
    for name in candidates:
        if name in df.columns:
            return str(name)
        found = lower.get(name.strip().lower())
        if found is not None:
            return str(found)
    return None


def _numeric_column(df: pd.DataFrame, candidates: tuple[str, ...], index: pd.Index) -> pd.Series:
    col = _find_column(df, candidates)
    if col is None:
        return pd.Series(np.nan, index=index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _string_column(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
    index: pd.Index,
    *,
    default: str = "",
) -> pd.Series:
    col = _find_column(df, candidates)
    if col is None:
        return pd.Series(default, index=index, dtype="object")
    return df[col].astype("string").fillna("").astype(str).str.strip()


def _camera_name_field_columns(
    raw: pd.DataFrame,
    index: pd.Index,
    camera: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    camera_name = _string_column(raw, ("camera_name", "camera name", "cam_name"), index)
    field = _string_column(raw, ("field", "asassn_field", "asassn field"), index)

    combined_col = _find_column(raw, ("camera_field", "cam_field", "camera,field", "cam/field"))
    field_only_combined_mask = pd.Series(False, index=index)
    if combined_col is not None:
        combined = raw[combined_col].astype("string").fillna("").astype(str).str.strip()
        split = combined.str.split("/", n=1, expand=True)
        if split.shape[1] >= 2:
            split_camera = split[0].fillna("").astype(str).str.strip()
            split_field = split[1].fillna("").astype(str).str.strip()
            slash_mask = combined.str.contains("/", regex=False).fillna(False)
            camera_name = camera_name.mask((camera_name == "") & slash_mask, split_camera)
            field = field.mask((field == "") & slash_mask, split_field)
        else:
            slash_mask = pd.Series(False, index=index)

        if str(combined_col).strip().lower() == "camera_field":
            field_only_combined_mask = ~slash_mask & (combined != "")
            field = field.mask((field == "") & field_only_combined_mask, combined)
    else:
        combined = pd.Series("", index=index, dtype="object")

    has_named_camera_col = _find_column(raw, ("camera_name", "camera name", "cam_name")) is not None
    has_numbered_camera_col = (
        _find_column(raw, ("camera#", "camera_id", "camera id", "camera number", "camera_number")) is not None
    )
    if not has_named_camera_col and not has_numbered_camera_col:
        camera_name = camera_name.mask(
            (camera_name == "") & ~field_only_combined_mask,
            camera.astype(str).str.strip(),
        )

    return camera_name, field


def _normalize_band_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    lower = text.lower()
    if lower in {"0", "g", "gp", "zg", "sloan-g", "sdss-g"}:
        return "g"
    if lower in {"1", "v", "johnson-v", "johnson_v"}:
        return "V"
    return text


def _band_from_legacy(value: object) -> str:
    try:
        intval = int(float(value))
    except Exception:
        return _normalize_band_value(value)
    return "g" if intval == 0 else "V" if intval == 1 else str(value)


def _legacy_band_code(value: object) -> float:
    text = _normalize_band_value(value)
    if text.lower() == "g":
        return 0.0
    if text.lower() == "v":
        return 1.0
    return np.nan


def _boolish_series(values: pd.Series, *, default: bool = False) -> pd.Series:
    out = pd.Series(default, index=values.index, dtype=bool)
    for idx, value in values.items():
        if value is None or pd.isna(value):
            out.at[idx] = default
            continue
        if isinstance(value, (bool, np.bool_)):
            out.at[idx] = bool(value)
            continue
        text = str(value).strip().upper()
        if text in _GOOD_STRINGS:
            out.at[idx] = True
        elif text in _BAD_STRINGS:
            out.at[idx] = False
        else:
            try:
                out.at[idx] = bool(float(text))
            except Exception:
                out.at[idx] = default
    return out


def _quality_from_raw(df: pd.DataFrame, index: pd.Index) -> pd.Series:
    quality_col = _find_column(df, ("quality", "Quality", "quality_flag"))
    if quality_col is not None:
        return df[quality_col].astype("string").fillna("").astype(str).str.strip().str.upper()

    good_col = _find_column(df, ("good_bad", "good", "is_good"))
    if good_col is None:
        return pd.Series("G", index=index, dtype="object")

    if str(good_col).strip().lower() == "good_bad":
        numeric = pd.to_numeric(df[good_col], errors="coerce")
        if numeric.notna().any():
            values = set(numeric.dropna().astype(int).unique().tolist())
            if values in ({0}, {1}):
                good = pd.Series(True, index=index)
            elif values <= {0, 1}:
                good = numeric.eq(1)
            else:
                good = numeric.ne(0)
        else:
            good = _boolish_series(df[good_col], default=True)
    else:
        good = _boolish_series(df[good_col], default=True)
    return good.map(lambda value: "G" if bool(value) else "B")


def _source_path_series(df: pd.DataFrame, index: pd.Index, source_path: str | Path | None) -> pd.Series:
    if source_path is not None:
        return pd.Series(str(source_path), index=index, dtype="object")
    col = _find_column(df, ("source_path", "path", "lc_path"))
    if col is not None:
        return df[col].astype("string").fillna("").astype(str)
    return pd.Series("", index=index, dtype="object")


def _compute_mag_flux_density(out: pd.DataFrame) -> None:
    has_survey_flux = out["flux"].notna()
    out["flux_density_mjy"] = np.nan
    out["flux_density_mjy_err"] = np.nan
    out["flux_provenance"] = ""

    out.loc[has_survey_flux, "flux_density_mjy"] = out.loc[has_survey_flux, "flux"]
    out.loc[has_survey_flux, "flux_density_mjy_err"] = out.loc[has_survey_flux, "flux_err"]
    out.loc[has_survey_flux, "flux_provenance"] = "asassn_survey_flux"

    mag = pd.to_numeric(out["mag"], errors="coerce")
    mag_err = pd.to_numeric(out["mag_err"], errors="coerce")
    band = out["band"].astype(str).str.strip().str.lower()

    g_mask = ~has_survey_flux & mag.notna() & band.eq("g")
    v_mask = ~has_survey_flux & mag.notna() & band.eq("v")
    for mask, zero_jy, provenance in (
        (g_mask, AB_ZERO_POINT_JY, "mag_zero_point_g_ab"),
        (v_mask, V_VEGA_ZERO_POINT_JY, "mag_zero_point_v_vega"),
    ):
        if not bool(mask.any()):
            continue
        flux_mjy = zero_jy * 1000.0 * np.power(10.0, -0.4 * mag.loc[mask])
        out.loc[mask, "flux_density_mjy"] = flux_mjy
        out.loc[mask, "flux_density_mjy_err"] = MAG_ERROR_TO_FRAC_FLUX * flux_mjy * mag_err.loc[mask]
        out.loc[mask, "flux_provenance"] = provenance


def _compute_relative_flux(out: pd.DataFrame) -> None:
    out["rel_flux"] = np.nan
    out["rel_flux_err"] = np.nan
    mag = pd.to_numeric(out["mag"], errors="coerce")
    mag_err = pd.to_numeric(out["mag_err"], errors="coerce")

    if mag.notna().any():
        band_key = out["band"].astype(str).replace("", "all")
        for _, idx in band_key.groupby(band_key).groups.items():
            idx = pd.Index(idx)
            med_mag = float(np.nanmedian(mag.loc[idx]))
            if not np.isfinite(med_mag):
                continue
            rel = np.power(10.0, -0.4 * (mag.loc[idx] - med_mag))
            out.loc[idx, "rel_flux"] = rel
            out.loc[idx, "rel_flux_err"] = MAG_ERROR_TO_FRAC_FLUX * rel * mag_err.loc[idx]
        return

    flux_density = pd.to_numeric(out["flux_density_mjy"], errors="coerce")
    if not flux_density.notna().any():
        return
    band_key = out["band"].astype(str).replace("", "all")
    for _, idx in band_key.groupby(band_key).groups.items():
        idx = pd.Index(idx)
        values = flux_density.loc[idx]
        finite_positive = values[np.isfinite(values) & (values > 0)]
        if finite_positive.empty:
            continue
        scale = float(np.nanmedian(finite_positive))
        out.loc[idx, "rel_flux"] = values / scale
        out.loc[idx, "rel_flux_err"] = pd.to_numeric(out.loc[idx, "flux_density_mjy_err"], errors="coerce") / scale


def _read_lightcurve_raw(path: str | Path, *, file_ext: str | None = None) -> pd.DataFrame:
    from malca.config import LIGHT_CURVE_FILE_EXTENSION

    lc_path = Path(path)
    suffix = lc_path.suffix.lower()

    if file_ext is None and suffix:
        file_ext = suffix[1:] if suffix.startswith(".") else suffix
    elif file_ext is None:
        file_ext = LIGHT_CURVE_FILE_EXTENSION
    if file_ext.startswith("."):
        file_ext = file_ext[1:]

    if file_ext in ("dat2", "dat3", "dat"):
        dfg, dfv = read_lc_dat2(lc_path.stem, str(lc_path.parent), file_ext=file_ext)
        if dfg.empty and dfv.empty:
            return pd.DataFrame()
        return pd.concat([dfg, dfv], ignore_index=True)
    if suffix == ".csv":
        return pd.read_csv(lc_path, comment="#", skip_blank_lines=True)
    return read_asassn_dat(lc_path)


def normalize_asassn_lightcurve(
    frame_or_path: pd.DataFrame | str | Path,
    *,
    apply_quality: bool = True,
    source_path: str | Path | None = None,
    sort_by_time: bool = True,
) -> pd.DataFrame:
    """Return a canonical lowercase ASAS-SN light-curve table.

    Survey-provided SkyPatrol flux remains in ``flux``/``flux_err``. Magnitude-only
    rows get band-zero-point estimates in ``flux_density_mjy`` and shape-only
    values in ``rel_flux``.
    """
    if isinstance(frame_or_path, (str, Path)):
        source_path = Path(frame_or_path)
        raw = _read_lightcurve_raw(source_path)
    else:
        raw = frame_or_path.copy()

    if raw is None or raw.empty:
        return _empty_canonical()

    index = raw.index
    out = pd.DataFrame(index=index)

    jd_col = _find_column(raw, ("jd", "JD", "hjd", "HJD", "bjd", "BJD"))
    mjd_col = _find_column(raw, ("mjd", "MJD"))
    if jd_col is not None:
        out["jd"] = pd.to_numeric(raw[jd_col], errors="coerce")
        finite_jd = out["jd"].dropna()
        if not finite_jd.empty and float(finite_jd.median()) < 50000.0:
            out["jd"] = out["jd"] + ASASSN_REDUCED_JD_OFFSET
        out["mjd"] = out["jd"] - MJD_OFFSET
    elif mjd_col is not None:
        out["mjd"] = pd.to_numeric(raw[mjd_col], errors="coerce")
        out["jd"] = out["mjd"] + MJD_OFFSET
    else:
        raise ValueError("light curve is missing a JD/MJD time column")

    band_col = _find_column(raw, ("band", "filter", "Filter", "phot_filter", "filter_band", "passband"))
    legacy_band_col = _find_column(raw, ("v_g_band",))
    if band_col is not None:
        out["band"] = raw[band_col].map(_normalize_band_value)
    elif legacy_band_col is not None:
        out["band"] = raw[legacy_band_col].map(_band_from_legacy)
    else:
        out["band"] = ""

    out["mag"] = _numeric_column(raw, ("mag", "Mag", "magnitude", "m"), index)
    out["mag_err"] = _numeric_column(raw, ("mag_err", "Mag Error", "mag error", "mag_error", "magerr", "error", "merr"), index)

    flux_col = _find_column(raw, ("flux", "Flux"))
    flux_err_col = _find_column(raw, ("flux_err", "Flux Error", "flux error", "flux_error"))
    if flux_col is not None:
        out["flux"] = pd.to_numeric(raw[flux_col], errors="coerce")
        out["flux_err"] = pd.to_numeric(raw[flux_err_col], errors="coerce") if flux_err_col else np.nan
    else:
        out["flux"] = np.nan
        out["flux_err"] = np.nan

    out["quality"] = _quality_from_raw(raw, index)
    out["camera"] = _string_column(raw, ("camera", "Camera", "camera#", "camera_name"), index)
    out["camera_name"], out["field"] = _camera_name_field_columns(raw, index, out["camera"])
    out["limit"] = _numeric_column(raw, ("limit", "Limit"), index)
    out["fwhm"] = _numeric_column(raw, ("fwhm", "FWHM"), index)
    out["source_path"] = _source_path_series(raw, index, source_path)

    saturated_col = _find_column(raw, ("saturated", "Saturated"))
    if saturated_col is None:
        out["saturated"] = False
    else:
        out["saturated"] = _boolish_series(raw[saturated_col], default=False)

    _compute_mag_flux_density(out)
    _compute_relative_flux(out)

    quality_good = _boolish_series(out["quality"], default=True)
    finite_time = out["jd"].notna()
    finite_value = out["mag"].notna() | out["flux"].notna() | out["flux_density_mjy"].notna() | out["rel_flux"].notna()
    mag_err_ok = pd.to_numeric(out["mag_err"], errors="coerce").gt(0)
    flux_err_ok = pd.to_numeric(out["flux_err"], errors="coerce").gt(0)
    flux_density_err_ok = pd.to_numeric(out["flux_density_mjy_err"], errors="coerce").gt(0)
    rel_flux_err_ok = pd.to_numeric(out["rel_flux_err"], errors="coerce").gt(0)
    uncertainty_ok = (
        (out["mag"].notna() & mag_err_ok)
        | (out["flux"].notna() & flux_err_ok)
        | (out["flux_density_mjy"].notna() & flux_density_err_ok)
        | (out["rel_flux"].notna() & rel_flux_err_ok)
    )
    out["is_good"] = quality_good & ~out["saturated"].astype(bool) & finite_time & finite_value & uncertainty_ok

    out = out[CANONICAL_ASASSN_COLUMNS]
    if apply_quality:
        out = out.loc[out["is_good"]].copy()
    if sort_by_time:
        out = out.sort_values("jd")
    return out.reset_index(drop=True)


def filter_asassn_quality(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply MALCA's centralized ASAS-SN row-quality filter."""
    return normalize_asassn_lightcurve(frame, apply_quality=True)


def to_asassn_algorithm_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert a canonical ASAS-SN frame to the working columns used by event algorithms."""
    canonical = normalize_asassn_lightcurve(frame, apply_quality=False, sort_by_time=False)
    out = pd.DataFrame(index=canonical.index)
    out["JD"] = pd.to_numeric(canonical["jd"], errors="coerce")
    out["mag"] = pd.to_numeric(canonical["mag"], errors="coerce")
    out["error"] = pd.to_numeric(canonical["mag_err"], errors="coerce")
    out["good_bad"] = canonical["is_good"].astype(int)
    out["camera#"] = canonical["camera"].astype(str)
    out["v_g_band"] = canonical["band"].map(_legacy_band_code)
    out["saturated"] = canonical["saturated"].astype(int)
    out["camera_name"] = canonical["camera_name"].astype(str)
    out["field"] = canonical["field"].astype(str)
    return out.reset_index(drop=True)


def load_lightcurve_df(
    path: str | Path,
    *,
    file_ext: str | None = None,
    filter_bad_cameras_enabled: bool = False,
    bad_camera_scatter_ratio: float = BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    apply_quality: bool = True,
    return_filtered_info: bool = False,
):
    """Load a native MALCA light curve as the canonical lowercase ASAS-SN schema."""
    lc_path = Path(path)
    df = _read_lightcurve_raw(lc_path, file_ext=file_ext)
    if df.empty:
        empty = _empty_canonical()
        return (empty, set()) if return_filtered_info else empty

    filtered_cameras: set[int] = set()
    if filter_bad_cameras_enabled and not df.empty and "camera#" in df.columns:
        df, filtered_cameras = filter_bad_cameras(
            df,
            lc_path=str(lc_path),
            scatter_ratio_threshold=bad_camera_scatter_ratio,
        )

    df = normalize_asassn_lightcurve(df, apply_quality=apply_quality, source_path=lc_path)
    if return_filtered_info:
        return df, filtered_cameras
    return df
