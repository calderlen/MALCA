from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np
import pandas as pd


AGN_TYPE_RE = re.compile(
    r"\b(?:agn|qso|quasar|seyfert|liner|blazar|bllac|blaq|broad[-\s]?line|type\s*1)\b",
    re.IGNORECASE,
)
SN_TYPE_RE = re.compile(r"\b(?:sn|supernova|snia|snii|slsn|cv|nova)\b", re.IGNORECASE)


def _num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _text(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series("", index=frame.index, dtype=object)
    return frame[col].fillna("").astype(str)


def _bool(frame: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[col]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default).astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").fillna(float(default)).astype(float) != 0.0
    lowered = values.fillna("").astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y", "match", "detected", "variable", "ok"})


def _clip01(values: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return np.clip(values, 0.0, 1.0)


def _finite_score(values: pd.Series, *, low: float, high: float) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    score = (vals - low) / (high - low)
    return score.clip(0.0, 1.0).fillna(0.0)


def _inverse_finite_score(values: pd.Series, *, low: float, high: float) -> pd.Series:
    return (1.0 - _finite_score(values, low=low, high=high)).clip(0.0, 1.0)


def _contains_any(frame: pd.DataFrame, cols: Iterable[str], pattern: re.Pattern[str]) -> pd.Series:
    out = pd.Series(False, index=frame.index, dtype=bool)
    for col in cols:
        if col in frame.columns:
            out |= frame[col].fillna("").astype(str).str.contains(pattern, na=False)
    return out


def _reason_join(parts: list[pd.Series]) -> pd.Series:
    if not parts:
        return pd.Series("", dtype=object)
    out = pd.Series("", index=parts[0].index, dtype=object)
    for part in parts:
        text = part.fillna("").astype(str)
        mask = text.str.strip().ne("")
        empty = out.eq("")
        out.loc[mask & empty] = text.loc[mask & empty]
        append = mask & ~empty
        out.loc[append] = out.loc[append] + "; " + text.loc[append]
    return out.str.strip("; ")


def _reason_if(mask: pd.Series, text: str) -> pd.Series:
    return pd.Series(np.where(mask.fillna(False), text, ""), index=mask.index, dtype=object)


def add_gaia_prior_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    parallax = _num(out, "parallax")
    parallax_error = _num(out, "parallax_error")
    parallax_snr = (parallax / parallax_error.replace(0, np.nan)).abs()
    if "gaia_parallax_snr" not in out.columns:
        out["gaia_parallax_snr"] = parallax_snr

    pm_total = _num(out, "pm_total")
    if pm_total.isna().all():
        pm_total = np.hypot(_num(out, "pmra").fillna(0.0), _num(out, "pmdec").fillna(0.0))
    pmra_error = _num(out, "pmra_error")
    pmdec_error = _num(out, "pmdec_error")
    pm_error = np.hypot(pmra_error, pmdec_error).replace(0, np.nan)
    pm_snr = (pm_total / pm_error).abs()
    pm_snr = pm_snr.where(pm_snr.notna(), _finite_score(pm_total, low=5.0, high=25.0) * 10.0)
    if "gaia_pm_snr" not in out.columns:
        out["gaia_pm_snr"] = pm_snr

    parallax_component = _finite_score(parallax_snr, low=3.0, high=8.0)
    pm_component = _finite_score(pm_snr, low=3.0, high=8.0)
    high_pm = _bool(out, "high_pm_flag")
    stellar = pd.concat([parallax_component, pm_component, high_pm.astype(float)], axis=1).max(axis=1)
    out["gaia_stellar_veto_score"] = stellar.clip(0.0, 1.0).fillna(0.0)

    has_astrometry = parallax.notna() | pm_total.notna() | _text(out, "gaia_id").str.strip().ne("") | _text(out, "source_id").str.strip().ne("")
    extragal = (1.0 - out["gaia_stellar_veto_score"]).clip(0.0, 1.0)
    extragal = extragal.where(has_astrometry, 0.5)
    agn_like_type = _contains_any(out, ("simbad_otype", "spectral_type", "host_spectral_class"), AGN_TYPE_RE)
    extragal = pd.Series(np.where(agn_like_type, np.maximum(extragal, 0.85), extragal), index=out.index)
    out["gaia_extragalactic_prior_score"] = extragal.clip(0.0, 1.0)
    return out


def add_wise_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    w1_w2 = _num(out, "w1_w2")
    if w1_w2.isna().all():
        w1_w2 = _num(out, "w1") - _num(out, "w2")
        out["w1_w2"] = w1_w2
    color_score = _finite_score(w1_w2, low=0.45, high=0.8)
    unwise_var = _bool(out, "unwise_w1_var") | (_num(out, "unwise_w1_zscore").abs() >= 5.0) | (_num(out, "unwise_w2_zscore").abs() >= 5.0)
    out["wise_agn_score"] = pd.concat([color_score, unwise_var.astype(float) * 0.65], axis=1).max(axis=1).clip(0.0, 1.0)

    w1_range = _num(out, "neowise_w1_range")
    w2_range = _num(out, "neowise_w2_range")
    n_epochs = _num(out, "neowise_n_epochs")
    range_score = pd.concat(
        [
            _finite_score(w1_range, low=0.15, high=0.7),
            _finite_score(w2_range, low=0.15, high=0.7),
        ],
        axis=1,
    ).max(axis=1)
    slope_score = pd.concat(
        [
            _finite_score(_num(out, "ltv_neowise_w1_slope").abs(), low=0.02, high=0.15),
            _finite_score(_num(out, "ltv_neowise_w1_w2_slope").abs(), low=0.01, high=0.08),
        ],
        axis=1,
    ).max(axis=1)
    epoch_penalty = _finite_score(n_epochs, low=2.0, high=5.0)
    out["neowise_variability_score"] = (pd.concat([range_score, slope_score], axis=1).max(axis=1) * epoch_penalty).clip(0.0, 1.0)
    return out


def add_multiwavelength_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["radio_agn_prior_score"] = pd.concat(
        [
            _bool(out, "radio_det").astype(float),
            _finite_score(_num(out, "radio_flux_mjy"), low=1.0, high=20.0),
        ],
        axis=1,
    ).max(axis=1).clip(0.0, 1.0)
    out["xray_agn_prior_score"] = pd.concat(
        [
            _bool(out, "xray_det").astype(float),
            _bool(out, "swift_xrt_det").astype(float),
            _finite_score(_num(out, "xray_flux"), low=1e-14, high=1e-12),
        ],
        axis=1,
    ).max(axis=1).clip(0.0, 1.0)

    galex_nuv = _num(out, "galex_nuv")
    galex_fuv = _num(out, "galex_fuv")
    optical_g = _num(out, "phot_g_mean_mag")
    nuv_g = galex_nuv - optical_g
    uv_color_score = _inverse_finite_score(nuv_g, low=0.0, high=3.0)
    uv_det_score = (galex_nuv.notna() | galex_fuv.notna()).astype(float) * 0.35
    swift_uv = _bool(out, "swift_uvot_det") | _bool(out, "swift_uvot_obs")
    out["uv_tde_score"] = pd.concat([uv_color_score, uv_det_score, swift_uv.astype(float) * 0.8], axis=1).max(axis=1).clip(0.0, 1.0)
    return out


def add_agn_prior_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    known_agn = (
        _bool(out, "ltv_milliquas_match")
        | _bool(out, "milliquas_match")
        | _bool(out, "prior_agn_spectrum_flag")
        | _bool(out, "broad_line_flag")
        | _bool(out, "known_clagn_match")
        | _contains_any(out, ("simbad_otype", "spectral_type", "host_spectral_class", "tns_type", "alerce_lc_class"), AGN_TYPE_RE)
    )
    components = pd.concat(
        [
            known_agn.astype(float),
            _num(out, "wise_agn_score", 0.0),
            _num(out, "neowise_variability_score", 0.0) * 0.85,
            _num(out, "xray_agn_prior_score", 0.0),
            _num(out, "radio_agn_prior_score", 0.0),
        ],
        axis=1,
    )
    out["agn_prior_score"] = components.max(axis=1).clip(0.0, 1.0)
    out["agn_prior_reasons"] = _reason_join(
        [
            _reason_if(known_agn, "known AGN/QSO/spectral AGN flag"),
            _reason_if(_num(out, "wise_agn_score", 0.0) >= 0.5, "WISE AGN-like color/variability"),
            _reason_if(_num(out, "neowise_variability_score", 0.0) >= 0.5, "NEOWISE mid-IR variability"),
            _reason_if(_num(out, "xray_agn_prior_score", 0.0) >= 0.5, "X-ray counterpart"),
            _reason_if(_num(out, "radio_agn_prior_score", 0.0) >= 0.5, "radio counterpart"),
        ]
    )
    return out


def add_tde_candidate_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    single_flare = _num(out, "tde_single_flare_score")
    if single_flare.isna().all():
        n_flares = _num(out, "n_flare_events")
        single_flare = pd.Series(np.where(n_flares.eq(1), 1.0, np.where(n_flares.gt(1), 0.0, np.nan)), index=out.index)
    single_flare = single_flare.fillna(_bool(out, "single_flare").astype(float) if "single_flare" in out.columns else 0.0)

    quiet = _num(out, "tde_quiet_baseline_score")
    if quiet.isna().all():
        quiet = _inverse_finite_score(_num(out, "preflare_rms"), low=0.03, high=0.25)
    quiet = quiet.fillna(0.0)

    no_recur = _num(out, "tde_no_recurrence_score")
    if no_recur.isna().all():
        recurrence = _num(out, "recurrence_count")
        no_recur = pd.Series(np.where(recurrence.eq(0), 1.0, np.where(recurrence.gt(0), 0.0, np.nan)), index=out.index)
    no_recur = no_recur.fillna(0.0)

    smooth = _num(out, "tde_smooth_decline_score")
    if smooth.isna().all():
        smooth = pd.concat(
            [
                _finite_score(_num(out, "fallback_fit_r2"), low=0.4, high=0.85),
                _finite_score(_num(out, "decline_smoothness"), low=0.3, high=0.8),
            ],
            axis=1,
        ).max(axis=1)
    smooth = smooth.fillna(0.0)

    nuclear = _num(out, "host_nuclear_score", 0.5).fillna(0.5)
    nonstellar = _num(out, "gaia_extragalactic_prior_score", 0.5).fillna(0.5)
    no_agn = (1.0 - _num(out, "agn_prior_score", 0.0).fillna(0.0)).clip(0.0, 1.0)
    no_sn = (~(_contains_any(out, ("tns_type", "alerce_lc_class", "known_class"), SN_TYPE_RE))).astype(float)
    uv = _num(out, "uv_tde_score", 0.0).fillna(0.0)
    z_known = _num(out, "redshift").notna().astype(float) * 0.35

    score = (
        0.20 * single_flare
        + 0.16 * quiet
        + 0.16 * no_recur
        + 0.14 * smooth
        + 0.14 * nuclear
        + 0.10 * nonstellar
        + 0.06 * no_agn
        + 0.02 * uv
        + 0.02 * z_known
    ) * no_sn
    out["tde_candidate_score"] = score.clip(0.0, 1.0)
    out["tde_candidate_reasons"] = _reason_join(
        [
            _reason_if(single_flare >= 0.7, "single flare"),
            _reason_if(quiet >= 0.7, "quiet pre-flare baseline"),
            _reason_if(no_recur >= 0.7, "no recurrence"),
            _reason_if(smooth >= 0.7, "smooth decline"),
            _reason_if(nuclear >= 0.7, "nuclear host association"),
            _reason_if(nonstellar >= 0.7, "Gaia nonstellar"),
            _reason_if(uv >= 0.5, "UV/Swift support"),
            _reason_if(_num(out, "agn_prior_score", 0.0) >= 0.5, "demoted by prior AGN evidence"),
            _reason_if(no_sn <= 0.0, "demoted by SN/CV/nova classification"),
        ]
    )
    return out


def add_clagn_photometric_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    state_change = pd.concat(
        [
            _finite_score(_num(out, "clagn_state_change_mag").abs(), low=0.5, high=2.0),
            _finite_score(_num(out, "science_flux_frac_amp_p95_p05"), low=0.3, high=2.0),
            _finite_score(_num(out, "nuc_flux_frac_amp_p95_p05"), low=0.3, high=2.0),
        ],
        axis=1,
    ).max(axis=1)
    monotonic = pd.concat(
        [
            _finite_score(_num(out, "clagn_monotonicity_score"), low=0.4, high=0.9),
            _finite_score(_num(out, "science_flux_slope_snr").abs(), low=3.0, high=10.0),
            _finite_score(_num(out, "nuc_flux_slope_snr").abs(), low=3.0, high=10.0),
        ],
        axis=1,
    ).max(axis=1)
    plateau = _num(out, "clagn_plateau_score").fillna(_bool(out, "new_long_lived_state").astype(float) if "new_long_lived_state" in out.columns else 0.0)
    agn = _num(out, "agn_prior_score", 0.0).fillna(0.0)
    mid_ir = _num(out, "neowise_variability_score", 0.0).fillna(0.0)
    spectrum = (_bool(out, "broad_line_change_flag") | _bool(out, "known_clagn_match")).astype(float)
    score = 0.26 * state_change + 0.18 * monotonic + 0.16 * plateau + 0.20 * agn + 0.10 * mid_ir + 0.10 * spectrum
    out["clagn_photometric_score"] = score.clip(0.0, 1.0)
    out["clagn_reasons"] = _reason_join(
        [
            _reason_if(state_change >= 0.6, "large optical state change"),
            _reason_if(monotonic >= 0.6, "monotonic long-term transition"),
            _reason_if(plateau >= 0.6, "new long-lived state"),
            _reason_if(agn >= 0.5, "prior AGN evidence"),
            _reason_if(mid_ir >= 0.5, "mid-IR variability/echo support"),
            _reason_if(spectrum > 0.0, "known/spectral CLAGN evidence"),
        ]
    )
    return out


def score_nuclear_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Add explainable AGN/TDE/CLAGN context scores to a candidate table."""
    out = add_gaia_prior_scores(df)
    out = add_wise_scores(out)
    out = add_multiwavelength_scores(out)
    out = add_agn_prior_score(out)
    out = add_tde_candidate_score(out)
    out = add_clagn_photometric_score(out)
    return out
