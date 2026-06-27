"""LTV-specific long-term summaries for external light-curve products."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from malca.config import GAIA_TCB_EPOCH_JD, KEPLER_BKJD_OFFSET, MJD_TO_JD, SKYPATROL_JD_OFFSET, TESS_BTJD_OFFSET
from malca.io.table_io import read_parquet_table, write_feature_table


LTV_MS_FEATURE_VERSION = "1"

LTV_MS_FEATURE_COLUMN_SPECS: tuple[tuple[str, str, str], ...] = (
    ("ltv_ms_feature_status", "TEXT", "text"),
    ("ltv_ms_feature_version", "TEXT", "text"),
    ("ltv_ms_atlas_n_points", "INTEGER", "float"),
    ("ltv_ms_atlas_time_span_days", "REAL", "float"),
    ("ltv_ms_atlas_mag_range", "REAL", "float"),
    ("ltv_ms_atlas_mag_median", "REAL", "float"),
    ("ltv_ms_atlas_mag_slope_per_year", "REAL", "float"),
    ("ltv_ms_ztf_n_points", "INTEGER", "float"),
    ("ltv_ms_ztf_time_span_days", "REAL", "float"),
    ("ltv_ms_ztf_mag_range", "REAL", "float"),
    ("ltv_ms_ztf_mag_median", "REAL", "float"),
    ("ltv_ms_ztf_mag_slope_per_year", "REAL", "float"),
    ("ltv_ms_gaia_epoch_n_points", "INTEGER", "float"),
    ("ltv_ms_gaia_epoch_time_span_days", "REAL", "float"),
    ("ltv_ms_gaia_epoch_mag_range", "REAL", "float"),
    ("ltv_ms_gaia_epoch_mag_median", "REAL", "float"),
    ("ltv_ms_gaia_epoch_mag_slope_per_year", "REAL", "float"),
    ("ltv_ms_neowise_n_points", "INTEGER", "float"),
    ("ltv_ms_neowise_time_span_days", "REAL", "float"),
    ("ltv_ms_neowise_w1_range", "REAL", "float"),
    ("ltv_ms_neowise_w1_median", "REAL", "float"),
    ("ltv_ms_neowise_w1_slope_per_year", "REAL", "float"),
    ("ltv_ms_neowise_w2_range", "REAL", "float"),
    ("ltv_ms_neowise_w2_median", "REAL", "float"),
    ("ltv_ms_neowise_w2_slope_per_year", "REAL", "float"),
    ("ltv_ms_allwise_mep_n_points", "INTEGER", "float"),
    ("ltv_ms_allwise_mep_time_span_days", "REAL", "float"),
    ("ltv_ms_allwise_mep_w1_range", "REAL", "float"),
    ("ltv_ms_allwise_mep_w1_median", "REAL", "float"),
    ("ltv_ms_allwise_mep_w1_slope_per_year", "REAL", "float"),
    ("ltv_ms_allwise_mep_w2_range", "REAL", "float"),
    ("ltv_ms_allwise_mep_w2_median", "REAL", "float"),
    ("ltv_ms_allwise_mep_w2_slope_per_year", "REAL", "float"),
    ("ltv_ms_allwise_mep_w3_range", "REAL", "float"),
    ("ltv_ms_allwise_mep_w3_median", "REAL", "float"),
    ("ltv_ms_allwise_mep_w3_slope_per_year", "REAL", "float"),
    ("ltv_ms_allwise_mep_w4_range", "REAL", "float"),
    ("ltv_ms_allwise_mep_w4_median", "REAL", "float"),
    ("ltv_ms_allwise_mep_w4_slope_per_year", "REAL", "float"),
    ("ltv_ms_tess_n_points", "INTEGER", "float"),
    ("ltv_ms_tess_time_span_days", "REAL", "float"),
    ("ltv_ms_tess_flux_range", "REAL", "float"),
    ("ltv_ms_tess_flux_median", "REAL", "float"),
    ("ltv_ms_tess_flux_slope_per_year", "REAL", "float"),
    ("ltv_ms_kepler_n_points", "INTEGER", "float"),
    ("ltv_ms_kepler_time_span_days", "REAL", "float"),
    ("ltv_ms_kepler_flux_range", "REAL", "float"),
    ("ltv_ms_kepler_flux_median", "REAL", "float"),
    ("ltv_ms_kepler_flux_slope_per_year", "REAL", "float"),
    ("ltv_ms_aavso_n_points", "INTEGER", "float"),
    ("ltv_ms_aavso_time_span_days", "REAL", "float"),
    ("ltv_ms_aavso_mag_range", "REAL", "float"),
    ("ltv_ms_aavso_mag_median", "REAL", "float"),
    ("ltv_ms_aavso_mag_slope_per_year", "REAL", "float"),
    ("ltv_ms_ogle_n_points", "INTEGER", "float"),
    ("ltv_ms_ogle_time_span_days", "REAL", "float"),
    ("ltv_ms_ogle_mag_range", "REAL", "float"),
    ("ltv_ms_ogle_mag_median", "REAL", "float"),
    ("ltv_ms_ogle_mag_slope_per_year", "REAL", "float"),
    ("ltv_ms_stripe82_n_points", "INTEGER", "float"),
    ("ltv_ms_stripe82_time_span_days", "REAL", "float"),
    ("ltv_ms_stripe82_mag_range", "REAL", "float"),
    ("ltv_ms_stripe82_mag_median", "REAL", "float"),
    ("ltv_ms_stripe82_mag_slope_per_year", "REAL", "float"),
    ("ltv_ms_vvvx_virac_n_points", "INTEGER", "float"),
    ("ltv_ms_vvvx_virac_time_span_days", "REAL", "float"),
    ("ltv_ms_vvvx_virac_z_range", "REAL", "float"),
    ("ltv_ms_vvvx_virac_z_median", "REAL", "float"),
    ("ltv_ms_vvvx_virac_z_slope_per_year", "REAL", "float"),
    ("ltv_ms_vvvx_virac_y_range", "REAL", "float"),
    ("ltv_ms_vvvx_virac_y_median", "REAL", "float"),
    ("ltv_ms_vvvx_virac_y_slope_per_year", "REAL", "float"),
    ("ltv_ms_vvvx_virac_j_range", "REAL", "float"),
    ("ltv_ms_vvvx_virac_j_median", "REAL", "float"),
    ("ltv_ms_vvvx_virac_j_slope_per_year", "REAL", "float"),
    ("ltv_ms_vvvx_virac_h_range", "REAL", "float"),
    ("ltv_ms_vvvx_virac_h_median", "REAL", "float"),
    ("ltv_ms_vvvx_virac_h_slope_per_year", "REAL", "float"),
    ("ltv_ms_vvvx_virac_ks_range", "REAL", "float"),
    ("ltv_ms_vvvx_virac_ks_median", "REAL", "float"),
    ("ltv_ms_vvvx_virac_ks_slope_per_year", "REAL", "float"),
    ("ltv_ms_ps1_n_points", "INTEGER", "float"),
    ("ltv_ms_ps1_time_span_days", "REAL", "float"),
    ("ltv_ms_ps1_mag_range", "REAL", "float"),
    ("ltv_ms_ps1_mag_median", "REAL", "float"),
    ("ltv_ms_ps1_mag_slope_per_year", "REAL", "float"),
    ("ltv_ms_crts_n_points", "INTEGER", "float"),
    ("ltv_ms_crts_time_span_days", "REAL", "float"),
    ("ltv_ms_crts_mag_range", "REAL", "float"),
    ("ltv_ms_crts_mag_median", "REAL", "float"),
    ("ltv_ms_crts_mag_slope_per_year", "REAL", "float"),
)

LTV_MS_FEATURE_COLUMNS: tuple[str, ...] = tuple(col for col, _sql, _kind in LTV_MS_FEATURE_COLUMN_SPECS)


_SURVEY_PREFIXES = {
    "atlas": "atlas_lc",
    "ztf": "ztf_lc",
    "gaia_epoch": "gaia_epoch_lc",
    "neowise": "neowise_lc",
    "allwise_mep": "allwise_mep_lc",
    "tess": "tess_lc",
    "kepler": "kepler_lc",
    "aavso": "aavso_lc",
    "ogle": "ogle_lc",
    "stripe82": "stripe82_lc",
    "vvvx_virac": "vvvx_virac_lc",
    "ps1": "ps1_lc",
    "crts": "crts_lc",
}


def _empty_feature_row() -> dict[str, object]:
    row: dict[str, object] = {col: np.nan for col in LTV_MS_FEATURE_COLUMNS}
    row["ltv_ms_feature_status"] = "no_external_lcs"
    row["ltv_ms_feature_version"] = LTV_MS_FEATURE_VERSION
    for col in LTV_MS_FEATURE_COLUMNS:
        if col.endswith("_n_points"):
            row[col] = 0
    return row


def _candidate_id(row: pd.Series | dict[str, Any]) -> str:
    for key in ("candidate_id", "asas_sn_id"):
        value = row.get(key) if isinstance(row, dict) else row.get(key)
        if pd.notna(value) and str(value).strip():
            text = str(value).strip()
            return text if key == "candidate_id" or text.startswith("ltv_") else f"ltv_{text}"
    return ""


def _read_external_lc(external_lc_dir: Path, survey: str, candidate_id: str) -> pd.DataFrame:
    path = external_lc_dir / f"{_SURVEY_PREFIXES[survey]}_{candidate_id}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        return read_parquet_table(path)
    except Exception:
        return pd.DataFrame()


def _first_existing_col(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _numeric_col(df: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    col = _first_existing_col(df, names)
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _time_jd(df: pd.DataFrame, survey: str) -> pd.Series:
    if "jd" in df.columns:
        time = pd.to_numeric(df["jd"], errors="coerce")
        median = float(time.median()) if time.notna().any() else np.nan
        if np.isfinite(median) and median > 2_400_000:
            return time
        return time + SKYPATROL_JD_OFFSET
    if "mjd" in df.columns:
        return pd.to_numeric(df["mjd"], errors="coerce") + MJD_TO_JD

    col = _first_existing_col(df, ("time", "Time", "hjd", "HJD", "obsTime"))
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    time = pd.to_numeric(df[col], errors="coerce")
    median = float(time.median()) if time.notna().any() else np.nan
    if not np.isfinite(median):
        return time
    if median > 2_400_000:
        return time
    if median > 40_000:
        return time + MJD_TO_JD
    if survey == "tess":
        return time + TESS_BTJD_OFFSET
    if survey == "kepler":
        return time + KEPLER_BKJD_OFFSET
    if survey == "gaia_epoch":
        return time + GAIA_TCB_EPOCH_JD
    return time + SKYPATROL_JD_OFFSET


def _slope_per_year(time_jd: pd.Series, value: pd.Series) -> float:
    valid = time_jd.notna() & value.notna()
    if int(valid.sum()) < 2:
        return np.nan
    x = time_jd.loc[valid].astype(float).to_numpy()
    y = value.loc[valid].astype(float).to_numpy()
    if not np.isfinite(x).all() or not np.isfinite(y).all() or float(np.ptp(x)) <= 0:
        return np.nan
    x_years = (x - float(np.nanmedian(x))) / 365.25
    try:
        return float(np.polyfit(x_years, y, 1)[0])
    except Exception:
        return np.nan


def _summarize_value_series(prefix: str, df: pd.DataFrame, survey: str, value_name: str, value: pd.Series) -> dict[str, object]:
    time = _time_jd(df, survey)
    valid = value.notna()
    out = {
        f"ltv_ms_{prefix}_n_points": int(valid.sum()),
        f"ltv_ms_{prefix}_time_span_days": np.nan,
        f"ltv_ms_{prefix}_{value_name}_range": np.nan,
        f"ltv_ms_{prefix}_{value_name}_median": np.nan,
        f"ltv_ms_{prefix}_{value_name}_slope_per_year": np.nan,
    }
    if int(valid.sum()) == 0:
        return out
    values = value.loc[valid].astype(float)
    out[f"ltv_ms_{prefix}_{value_name}_median"] = float(values.median())
    if len(values) >= 2:
        out[f"ltv_ms_{prefix}_{value_name}_range"] = float(values.max() - values.min())
    finite_time = time.loc[valid].dropna()
    if len(finite_time) >= 2:
        out[f"ltv_ms_{prefix}_time_span_days"] = float(finite_time.max() - finite_time.min())
    out[f"ltv_ms_{prefix}_{value_name}_slope_per_year"] = _slope_per_year(time, value)
    return out


def _summarize_mag_survey(df: pd.DataFrame, survey: str) -> dict[str, object]:
    mag = _numeric_col(df, ("mag", "m", "Mag", "MAG", "magnitude", "mag_psf"))
    return _summarize_value_series(survey, df, survey, "mag", mag)


def _summarize_tess(df: pd.DataFrame) -> dict[str, object]:
    flux = _numeric_col(df, ("flux", "sap_flux", "pdcsap_flux"))
    return _summarize_value_series("tess", df, "tess", "flux", flux)


def _summarize_kepler(df: pd.DataFrame) -> dict[str, object]:
    flux = _numeric_col(df, ("flux", "sap_flux", "pdcsap_flux"))
    return _summarize_value_series("kepler", df, "kepler", "flux", flux)


def _summarize_wise_bands(prefix: str, df: pd.DataFrame, survey: str, bands: tuple[str, ...]) -> dict[str, object]:
    time = _time_jd(df, survey)
    values_by_band = {
        band: _numeric_col(df, (f"{band}mpro", f"{band}_mag", band, band.upper()))
        for band in bands
    }
    any_valid = pd.Series(False, index=df.index)
    for values in values_by_band.values():
        any_valid = any_valid | values.notna()
    n_points = int(any_valid.sum())
    finite_time = time.loc[any_valid].dropna()
    out = {
        f"ltv_ms_{prefix}_n_points": n_points,
        f"ltv_ms_{prefix}_time_span_days": float(finite_time.max() - finite_time.min()) if len(finite_time) >= 2 else np.nan,
    }
    for band, values in values_by_band.items():
        valid = values.dropna()
        out[f"ltv_ms_{prefix}_{band}_range"] = float(valid.max() - valid.min()) if len(valid) >= 2 else np.nan
        out[f"ltv_ms_{prefix}_{band}_median"] = float(valid.median()) if len(valid) else np.nan
        out[f"ltv_ms_{prefix}_{band}_slope_per_year"] = _slope_per_year(time, values)
    return out


def _summarize_neowise(df: pd.DataFrame) -> dict[str, object]:
    return _summarize_wise_bands("neowise", df, "neowise", ("w1", "w2"))


def _summarize_allwise_mep(df: pd.DataFrame) -> dict[str, object]:
    return _summarize_wise_bands("allwise_mep", df, "allwise_mep", ("w1", "w2", "w3", "w4"))


def _summarize_band_mag_survey(prefix: str, df: pd.DataFrame, survey: str, bands: tuple[str, ...]) -> dict[str, object]:
    time = _time_jd(df, survey)
    band_col = _first_existing_col(df, ("band", "filter", "Filter"))
    mag = _numeric_col(df, ("mag", "m", "Mag", "MAG", "magnitude", "mag_psf"))
    if band_col is None:
        return _summarize_value_series(prefix, df, survey, "mag", mag)

    band_values = df[band_col].astype(str).str.strip().str.lower()
    out: dict[str, object] = {
        f"ltv_ms_{prefix}_n_points": int(mag.notna().sum()),
        f"ltv_ms_{prefix}_time_span_days": np.nan,
    }
    valid_all = mag.notna()
    finite_time = time.loc[valid_all].dropna()
    if len(finite_time) >= 2:
        out[f"ltv_ms_{prefix}_time_span_days"] = float(finite_time.max() - finite_time.min())
    for band in bands:
        mask = (band_values == band.lower()) & mag.notna()
        valid = mag.loc[mask].dropna().astype(float)
        out[f"ltv_ms_{prefix}_{band.lower()}_range"] = float(valid.max() - valid.min()) if len(valid) >= 2 else np.nan
        out[f"ltv_ms_{prefix}_{band.lower()}_median"] = float(valid.median()) if len(valid) else np.nan
        band_value = pd.Series(np.nan, index=df.index, dtype=float)
        band_value.loc[mask] = mag.loc[mask]
        out[f"ltv_ms_{prefix}_{band.lower()}_slope_per_year"] = _slope_per_year(time, band_value)
    return out


def compute_ltv_multi_survey_features(
    df: pd.DataFrame,
    *,
    external_lc_dir: Path | str | None,
) -> pd.DataFrame:
    """Append LTV-specific long-baseline external-survey summary columns."""
    out = df.copy()
    root = Path(external_lc_dir).expanduser() if external_lc_dir is not None else None
    rows: list[dict[str, object]] = []

    for _, row in out.iterrows():
        features = _empty_feature_row()
        candidate_id = _candidate_id(row)
        if root is None or not candidate_id:
            rows.append(features)
            continue

        any_data = False
        for survey in ("atlas", "ztf", "gaia_epoch", "ps1", "crts"):
            lc = _read_external_lc(root, survey, candidate_id)
            if lc.empty:
                continue
            summary = _summarize_mag_survey(lc, survey)
            features.update(summary)
            any_data = any_data or int(summary.get(f"ltv_ms_{survey}_n_points") or 0) > 0

        neowise = _read_external_lc(root, "neowise", candidate_id)
        if not neowise.empty:
            summary = _summarize_neowise(neowise)
            features.update(summary)
            any_data = any_data or int(summary.get("ltv_ms_neowise_n_points") or 0) > 0

        tess = _read_external_lc(root, "tess", candidate_id)
        if not tess.empty:
            summary = _summarize_tess(tess)
            features.update(summary)
            any_data = any_data or int(summary.get("ltv_ms_tess_n_points") or 0) > 0

        kepler = _read_external_lc(root, "kepler", candidate_id)
        if not kepler.empty:
            summary = _summarize_kepler(kepler)
            features.update(summary)
            any_data = any_data or int(summary.get("ltv_ms_kepler_n_points") or 0) > 0

        allwise_mep = _read_external_lc(root, "allwise_mep", candidate_id)
        if not allwise_mep.empty:
            summary = _summarize_allwise_mep(allwise_mep)
            features.update(summary)
            any_data = any_data or int(summary.get("ltv_ms_allwise_mep_n_points") or 0) > 0

        for survey in ("aavso", "ogle", "stripe82"):
            lc = _read_external_lc(root, survey, candidate_id)
            if lc.empty:
                continue
            summary = _summarize_mag_survey(lc, survey)
            features.update(summary)
            any_data = any_data or int(summary.get(f"ltv_ms_{survey}_n_points") or 0) > 0

        vvvx_virac = _read_external_lc(root, "vvvx_virac", candidate_id)
        if not vvvx_virac.empty:
            summary = _summarize_band_mag_survey("vvvx_virac", vvvx_virac, "vvvx_virac", ("z", "y", "j", "h", "ks"))
            features.update(summary)
            any_data = any_data or int(summary.get("ltv_ms_vvvx_virac_n_points") or 0) > 0

        if any_data:
            features["ltv_ms_feature_status"] = "ok"
        rows.append(features)

    features_df = pd.DataFrame(rows, index=out.index)
    for col in LTV_MS_FEATURE_COLUMNS:
        out[col] = features_df[col] if col in features_df.columns else np.nan
    return out


def write_ltv_multi_survey_features(
    df: pd.DataFrame,
    output_path: Path,
    *,
    external_lc_dir: Path | str | None,
) -> pd.DataFrame:
    out = compute_ltv_multi_survey_features(df, external_lc_dir=external_lc_dir)
    write_feature_table(out, output_path)
    return out
