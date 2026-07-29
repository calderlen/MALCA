"""
Outputs:
- Core timing/cadence stats (including 3-day exposure metrics and largest gaps)
- Photometric stats (weighted/unweighted/clipped/MAD/IQR/weighted std/percentiles)
- Quality & error stats (SNR dist, fractions by good/saturated)
- Variability diagnostics (reduced chisq, inverse von Neumann ratio, RoMS, lag-1 autocorr, trend slope, Stetson I/J/K/L + J(time)/L(time), Cody Q/M)
- Optional Lomb-Scargle periodogram summary stats
- Nightly/seasonal coverage & duty cycle
- Per-camera / per-field / per-band usage + offsets and scatter
"""
from collections import OrderedDict
from pathlib import Path
import sys, io, argparse, math

from astropy.timeseries import LombScargle
from celerite2 import GaussianProcess as _GP, terms as _cterms
from iar.IARModel import IARphikalman as _IARphikalman
from scipy import stats as sp_stats
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
import numpy as np
import pandas as pd

from malca.core.baseline import per_camera_gp_baseline
from malca.core.period_arbitration import (
    NATIVE_PERIOD_WITH_MULTIPLES_FACTORS,
    choose_native_harmonic_candidate,
    native_harmonic_period_candidates,
    period_alias_matches,
)
from malca.config import (
    MAD_SCALE,
    STETSON_PAIR_MAX_DT_DAYS,
    STETSON_REWEIGHT_A,
    STETSON_REWEIGHT_B,
    STETSON_REWEIGHT_MIN_ITERS,
    STETSON_REWEIGHT_MAX_ITERS,
    STETSON_REWEIGHT_RTOL,
    SNR_CONVERSION_FACTOR,
    LS_MIN_FREQUENCY,
    LS_MAX_FREQUENCY,
    LS_ALIAS_TOLERANCE,
    LS_ALIAS_PERIODS,
    PDM_PLAVCHAN_PHASE_WIDTH,
    PDM_PLAVCHAN_MIN_NEIGHBORS,
    PERIODOGRAM_REFINE_TOP_K,
    PERIODOGRAM_REFINE_WINDOW_STEPS,
    PERIODOGRAM_REFINE_N_GRID,
)
from malca.core.derived_stats import compute_derived_feature_row
from malca.products.feature_layers import to_layer_first_frame
from malca.core.periodogram import pdm_find_period, ce_find_period
from malca.core.utils import read_lc_dat2, read_lc_csv, read_skypatrol_lc_csv, compute_camera_loo_metrics, compute_field_summary


_LC_COLUMNS = [
    "JD",
    "mag",
    "error",
    "good_bad",
    "camera#",
    "v_g_band",
    "saturated",
    "camera_name",
    "field",
]

Q_TEMPLATE_METHOD = "phase_template_med500m2"
Q_TEMPLATE_EVALUATION = "cycle_block_out_of_fold_v1"
Q_TEMPLATE_N_PHASE_BINS = 500
Q_TEMPLATE_MIN_BIN_POINTS = 2
Q_TEMPLATE_SMOOTH_WINDOW_BINS = 1
Q_TEMPLATE_MIN_BIN_COVERAGE = 0.10
Q_TEMPLATE_NOISE_SUBTRACT = False
Q_PERIOD_ARBITRATION_FACTORS = NATIVE_PERIOD_WITH_MULTIPLES_FACTORS
Q_PERIOD_ARBITRATION_MIN_REL_IMPROVEMENT = 0.0
Q_PERIOD_ARBITRATION_UPWARD_MIN_REL_IMPROVEMENT = 0.0




# helpers
def weighted_mean(x, w):
    w = np.asarray(w, float)
    x = np.asarray(x, float)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return np.nan, np.nan
    w = w[mask]; x = x[mask]
    mu = np.sum(w * x) / np.sum(w)
    var = np.sum(w * (x - mu)**2) / np.sum(w)
    # Standard error of weighted mean (approx)
    sem = math.sqrt(1.0 / np.sum(w))
    return mu, sem

def robust_sigma(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return MAD_SCALE * np.median(np.abs(x - np.median(x)))


def three_sigma_clipped_mean_mag(mag) -> float:
    """Return the 3-sigma-about-median mean used by the stats summary."""
    values = np.asarray(mag, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    median = float(np.nanmedian(values))
    sigma = robust_sigma(values)
    if np.isfinite(sigma) and sigma > 0:
        values = values[np.abs(values - median) <= 3.0 * sigma]
    return float(np.nanmean(values)) if values.size else np.nan


def _prepare_stats_lightcurve_frame(raw: pd.DataFrame | None) -> pd.DataFrame:
    """Normalize loaded ASAS-SN light-curve columns for compute_stats."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=_LC_COLUMNS)

    df = raw.copy()
    df.columns = _LC_COLUMNS[:len(df.columns)] + [f"extra_{i}" for i in range(len(df.columns) - len(_LC_COLUMNS))]
    for col in _LC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    for col in ("JD", "mag", "error", "good_bad", "camera#", "v_g_band", "saturated"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["good_bad"] = df["good_bad"].fillna(1)
    df["saturated"] = df["saturated"].fillna(0)
    return df.dropna(subset=["JD", "mag", "error"]).sort_values("JD").reset_index(drop=True)


def _load_stats_lightcurve_frames(
    asassn_id: object,
    path: str | Path,
    *,
    file_ext: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one stats light curve without discarding an exact input path.

    Directory inputs retain the legacy ``<source_id>.<extension>`` lookup.  An
    existing file input is authoritative: its complete stem and its own suffix
    are used, so catalogue IDs containing punctuation are never truncated and a
    mixed-extension manifest does not accidentally load a same-named neighbor.
    """
    input_path = Path(path).expanduser()
    if input_path.is_file():
        exact_source_id = input_path.stem
        exact_extension = input_path.suffix.lstrip(".") or file_ext
        if input_path.suffix.lower() == ".csv":
            # The two supported CSV readers currently accept a directory plus
            # source name.  Using the exact file's full stem reconstructs only
            # the path we have already established, never a shortened ID.
            df_g, df_v = read_lc_csv(exact_source_id, str(input_path.parent))
            if df_g.empty and df_v.empty:
                df_g, df_v = read_skypatrol_lc_csv(
                    exact_source_id,
                    str(input_path.parent),
                )
            return df_g, df_v
        return read_lc_dat2(
            exact_source_id,
            str(input_path),
            file_ext=exact_extension,
        )

    df_g, df_v = read_lc_csv(asassn_id, path)
    if df_g.empty and df_v.empty:
        df_g, df_v = read_skypatrol_lc_csv(asassn_id, path)
    if df_g.empty and df_v.empty:
        df_g, df_v = read_lc_dat2(asassn_id, path, file_ext=file_ext)
    return df_g, df_v


def _filter_stats_lightcurve_frame(
    df: pd.DataFrame,
    *,
    use_only_good: bool,
    drop_dupes: bool,
    duplicate_subset: tuple[str, ...],
) -> pd.DataFrame:
    """Apply the basic compute_stats quality filters."""
    out = df.copy()
    if out.empty:
        return out
    if drop_dupes:
        subset = [col for col in duplicate_subset if col in out.columns]
        if subset:
            out = out[~out.duplicated(subset=subset, keep="first")].reset_index(drop=True)
    if use_only_good:
        good = pd.to_numeric(out.get("good_bad"), errors="coerce").fillna(1) == 1
        saturated = pd.to_numeric(out.get("saturated"), errors="coerce").fillna(0) == 0
        out = out[good & saturated].reset_index(drop=True)
    return out


def _align_v_to_g_with_overlap_policy(
    df: pd.DataFrame,
    *,
    min_points_per_band: int = 5,
    min_overlap_fraction: float = 0.5,
) -> tuple[pd.DataFrame, float, str]:
    """Align bands only when their observing windows substantially overlap."""
    if df.empty or "v_g_band" not in df or "JD" not in df or "mag" not in df:
        return df, np.nan, "none"
    band = pd.to_numeric(df["v_g_band"], errors="coerce")
    jd = pd.to_numeric(df["JD"], errors="coerce")
    mag = pd.to_numeric(df["mag"], errors="coerce")
    g = (band == 0) & jd.notna() & mag.notna()
    v = (band == 1) & jd.notna() & mag.notna()
    if int(g.sum()) < min_points_per_band or int(v.sum()) < min_points_per_band:
        return df, np.nan, "not_aligned_insufficient_band_points"

    g_lo, g_hi = float(jd[g].min()), float(jd[g].max())
    v_lo, v_hi = float(jd[v].min()), float(jd[v].max())
    overlap_lo, overlap_hi = max(g_lo, v_lo), min(g_hi, v_hi)
    g_span, v_span = g_hi - g_lo, v_hi - v_lo
    overlap = max(0.0, overlap_hi - overlap_lo)
    reference_span = max(min(g_span, v_span), 1e-12)
    overlap_fraction = overlap / reference_span
    if overlap <= 0 or overlap_fraction < float(min_overlap_fraction):
        return df, np.nan, "not_aligned_no_temporal_overlap"

    # For nearly coextensive surveys the full medians are the most stable
    # overlap estimate. For partial overlap, restrict the color estimate to
    # the common time interval so secular evolution is not absorbed as color.
    if overlap_fraction >= 0.9:
        g_ref, v_ref = g, v
    else:
        in_overlap = jd.between(overlap_lo, overlap_hi, inclusive="both")
        g_ref, v_ref = g & in_overlap, v & in_overlap
    if int(g_ref.sum()) < min_points_per_band or int(v_ref.sum()) < min_points_per_band:
        return df, np.nan, "not_aligned_insufficient_overlap_points"
    offset = float(np.median(mag[v_ref]) - np.median(mag[g_ref]))
    if not np.isfinite(offset):
        return df, np.nan, "not_aligned_invalid_overlap_offset"
    out = df.copy()
    out["mag_raw"] = mag
    out.loc[v, "mag"] = mag[v] - offset
    # Keep the established label for downstream schema compatibility; unlike
    # the former implementation, this label now guarantees temporal overlap.
    return out, offset, "v_median_to_g_median"


def _camera_band_normalized_q_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time, offset-normalized magnitude, and error arrays for Q."""
    if df.empty:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    work = df.copy()
    work["_q_time"] = pd.to_numeric(work.get("JD"), errors="coerce")
    work["_q_mag"] = pd.to_numeric(work.get("mag"), errors="coerce")
    work["_q_err"] = pd.to_numeric(work.get("error"), errors="coerce")

    camera = (
        work["camera_name"].astype("string")
        if "camera_name" in work.columns
        else work.get("camera#", pd.Series("unknown", index=work.index)).astype("string")
    )
    camera = camera.fillna("").str.strip()
    camera = camera.where(camera != "", "unknown")
    band = pd.to_numeric(work.get("v_g_band"), errors="coerce").astype("Float64").astype("string").fillna("unknown")
    work["_q_group"] = band + "|" + camera

    finite = (
        np.isfinite(work["_q_time"].to_numpy(dtype=float))
        & np.isfinite(work["_q_mag"].to_numpy(dtype=float))
        & np.isfinite(work["_q_err"].to_numpy(dtype=float))
        & (work["_q_err"].to_numpy(dtype=float) > 0)
    )
    work = work.loc[finite].copy()
    if work.empty:
        empty = np.array([], dtype=float)
        return empty, empty, empty

    global_median = float(np.median(work["_q_mag"].to_numpy(dtype=float)))
    group_median = work.groupby("_q_group")["_q_mag"].transform("median")
    offsets = pd.to_numeric(group_median, errors="coerce").fillna(global_median).to_numpy(dtype=float)
    values = work["_q_mag"].to_numpy(dtype=float) - offsets + global_median
    return (
        work["_q_time"].to_numpy(dtype=float),
        values.astype(float),
        work["_q_err"].to_numpy(dtype=float),
    )


def _stetson_robust_mean(
    mag,
    err,
    *,
    a=STETSON_REWEIGHT_A,
    b=STETSON_REWEIGHT_B,
    min_iters=STETSON_REWEIGHT_MIN_ITERS,
    max_iters=STETSON_REWEIGHT_MAX_ITERS,
    rtol=STETSON_REWEIGHT_RTOL,
):
    """Iteratively reweighted mean used by the paper-style Stetson indices."""
    mag = np.asarray(mag, float)
    err = np.asarray(err, float)
    mask = np.isfinite(mag) & np.isfinite(err) & (err > 0)
    if mask.sum() == 0:
        return np.nan

    mag = mag[mask]
    err = err[mask]
    weights = 1.0 / np.square(err)
    mu, _ = weighted_mean(mag, weights)
    if not np.isfinite(mu) or mag.size < 2:
        return float(mu)

    scale = math.sqrt(mag.size / (mag.size - 1.0))
    tiny = np.finfo(float).tiny
    a = float(a)
    b = float(b)

    for idx in range(max(0, int(max_iters))):
        resid = np.abs(scale * (mag - mu) / err)
        if a > 0 and b > 0:
            factors = 1.0 / (1.0 + np.power(resid / a, b))
        else:
            factors = np.ones_like(resid, dtype=float)

        weights = np.maximum(weights * factors, tiny)
        new_mu, _ = weighted_mean(mag, weights)
        if not np.isfinite(new_mu):
            break

        if idx + 1 >= max(1, int(min_iters)) and math.isclose(new_mu, mu, rel_tol=rtol, abs_tol=rtol):
            mu = new_mu
            break

        mu = new_mu

    return float(mu)


def weighted_std(x, err):
    """Inverse-variance weighted standard deviation (Sokolovsky et al. 2017)."""
    x = np.asarray(x, float)
    err = np.asarray(err, float)
    mask = np.isfinite(x) & np.isfinite(err) & (err > 0)
    if mask.sum() < 2:
        return np.nan

    x = x[mask]
    err = err[mask]
    w = 1.0 / np.square(err)
    sum_w = float(np.sum(w))
    sum_w2 = float(np.sum(np.square(w)))
    denom = sum_w * sum_w - sum_w2
    if sum_w <= 0 or denom <= 0:
        return np.nan

    mu = float(np.sum(w * x) / sum_w)
    return float(np.sqrt((sum_w / denom) * np.sum(w * np.square(x - mu))))


def paper_iqr(x):
    """Interquartile range using the paper's median-of-halves definition."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan

    xs = np.sort(x)
    n = xs.size
    lower = xs[: n // 2]
    upper = xs[(n + 1) // 2 :]
    if lower.size == 0 or upper.size == 0:
        return np.nan
    return float(np.median(upper) - np.median(lower))


def flux_asymmetry_metric(mag) -> float:
    """Cody-style flux asymmetry metric on magnitudes.

    Positive values indicate variability dominated by high-magnitude fading
    events; negative values indicate low-magnitude brightening events.
    """
    mag = np.asarray(mag, float)
    mag = mag[np.isfinite(mag)]
    if mag.size < 10:
        return np.nan

    sigma = float(np.std(mag, ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        return np.nan

    decile_n = max(1, int(np.floor(0.1 * mag.size)))
    sorted_mag = np.sort(mag)
    decile_mean = float(np.mean(np.concatenate([sorted_mag[:decile_n], sorted_mag[-decile_n:]])))
    median = float(np.median(mag))
    return float((decile_mean - median) / sigma)


def log_gaussian(x, mu, sigma):
    """
    Log probability of Gaussian distribution.

    ln p = -1/2 * ((x-mu)/sigma)^2 - ln(sigma) - 1/2 ln(2pi)

    Parameters
    ----------
    x : array-like
        Values
    mu : array-like
        Mean(s)
    sigma : array-like
        Standard deviation(s)

    Returns
    -------
    log_prob : array
        Log probabilities
    """
    x = np.asarray(x, float)
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    sigma = np.clip(sigma, 1e-12, np.inf)
    z = (x - mu) / sigma
    return -0.5 * z**2 - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)

def median_dt(jd: np.ndarray) -> float:
    """Median of positive successive time gaps (days)."""
    jd = np.asarray(jd, float)
    jd = jd[np.isfinite(jd)]
    if jd.size < 2:
        return np.nan
    dt = np.diff(np.sort(jd))
    dt = dt[dt > 0]
    return float(np.median(dt)) if dt.size > 0 else np.nan

def bic(resid, err, n_params):
    """
    Bayesian Information Criterion for model selection.

    BIC = n * ln(sigma^2) + k * ln(n)

    where sigma^2 is the variance of residuals, k is the number of parameters,
    and n is the number of data points.

    Parameters
    ----------
    resid : array-like
        Residuals (observed - model)
    err : array-like
        Uncertainties
    n_params : int
        Number of model parameters

    Returns
    -------
    bic_value : float
        BIC value (lower is better)
    """
    resid = np.asarray(resid, float)
    err = np.asarray(err, float)
    mask = np.isfinite(resid) & np.isfinite(err) & (err > 0)
    if mask.sum() < n_params + 1:
        return np.nan
    resid = resid[mask]
    err = err[mask]
    n = len(resid)
    # Use the full Gaussian log-likelihood so BIC follows the standard form:
    # BIC = k * ln(n) - 2 * ln(L_max)
    # where ln(L_max) = sum log Gaussian(resid; 0, err)
    logp = log_gaussian(resid, 0.0, err)
    logL = float(np.nansum(logp))
    return float(n_params * np.log(n) - 2.0 * logL)

def pct(x, q):
    return float(np.nanpercentile(x, q)) if len(x) else np.nan

def reduced_chisq(y, yerr, model_value):
    y  = np.asarray(y, float)
    ye = np.asarray(yerr, float)
    m  = np.asarray(model_value, float)
    mask = np.isfinite(y) & np.isfinite(ye) & (ye > 0)
    if mask.sum() < 2:
        return np.nan
    y = y[mask]; ye = ye[mask]
    chi2 = np.sum(((y - m)/ye)**2)
    dof = y.size - 1
    return chi2 / dof if dof > 0 else np.nan

def von_neumann_ratio(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3: return np.nan
    diffs = np.diff(x)
    num = np.mean(diffs**2)
    den = np.var(x, ddof=1)
    return num/den if den > 0 else np.nan


def inverse_von_neumann_ratio(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan
    diffs = np.diff(x)
    mean_sq_succ_diff = np.mean(np.square(diffs))
    variance = np.var(x, ddof=1)
    return float(variance / mean_sq_succ_diff) if mean_sq_succ_diff > 0 else np.nan

def lag1_autocorr(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3: return np.nan
    x0 = x[:-1] - x[:-1].mean()
    x1 = x[1:]  - x[1:].mean()
    den = np.sqrt(np.sum(x0**2) * np.sum(x1**2))
    return float(np.sum(x0 * x1) / den) if den > 0 else np.nan


def roms_statistic(mag, err):
    """Robust median statistic (RoMS) from Sokolovsky et al. 2017."""
    mag = np.asarray(mag, float)
    err = np.asarray(err, float)
    mask = np.isfinite(mag) & np.isfinite(err) & (err > 0)
    if mask.sum() < 2:
        return np.nan

    mag = mag[mask]
    err = err[mask]
    med = float(np.median(mag))
    return float(np.sum(np.abs(mag - med) / err) / (mag.size - 1.0))


def sokolovsky_peak_to_peak_variability(mag, err):
    """Return the uncertainty-aware peak-to-peak variability statistic ``v``.

    This is the Sokolovsky et al. (2017) definition used by Bredall et al.
    (2020):

    ``v = (max(m - sigma) - min(m + sigma)) / (max(m - sigma) + min(m + sigma))``.

    The statistic is dimensionless and intentionally uses the error-adjusted
    extrema rather than the robust ALeRCE ``amplitude`` feature. Negative
    values are retained when the two extrema overlap within their uncertainties.
    """
    mag = np.asarray(mag, float)
    err = np.asarray(err, float)
    mask = np.isfinite(mag) & np.isfinite(err) & (err >= 0)
    if mask.sum() < 2:
        return np.nan

    lower = mag[mask] - err[mask]
    upper = mag[mask] + err[mask]
    faint_limit = float(np.max(lower))
    bright_limit = float(np.min(upper))
    denominator = faint_limit + bright_limit
    if not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return float((faint_limit - bright_limit) / denominator)


def compute_sokolovsky_peak_to_peak_summary(
    asassn_id,
    path,
    *,
    use_only_good: bool = True,
    drop_dupes: bool = True,
    file_ext: str | None = None,
    input_frame: pd.DataFrame | None = None,
):
    """Compute one Sokolovsky ``v`` from a median-offset combined light curve.

    All usable g observations are retained.  When V is present, every usable
    V magnitude is shifted by ``median(V) - median(g)`` before g and V are
    concatenated and the statistic is calculated.  This is intentionally the
    only Sokolovsky implementation used by the July 1 backfill and plot.
    """

    if input_frame is not None:
        loaded = input_frame.copy()
        if "v_g_band" in loaded.columns:
            band_values = pd.to_numeric(loaded["v_g_band"], errors="coerce")
            df_g_raw = loaded.loc[band_values != 1].copy()
            df_v_raw = loaded.loc[band_values == 1].copy()
        else:
            df_g_raw, df_v_raw = loaded, pd.DataFrame()
    else:
        df_g_raw, df_v_raw = _load_stats_lightcurve_frames(
            asassn_id,
            path,
            file_ext=file_ext,
        )

    df_g = _filter_stats_lightcurve_frame(
        _prepare_stats_lightcurve_frame(df_g_raw),
        use_only_good=use_only_good,
        drop_dupes=drop_dupes,
        duplicate_subset=("JD", "camera#", "camera_name", "field"),
    )
    df_v = _filter_stats_lightcurve_frame(
        _prepare_stats_lightcurve_frame(df_v_raw),
        use_only_good=use_only_good,
        drop_dupes=drop_dupes,
        duplicate_subset=("JD", "camera#", "camera_name", "field"),
    )
    if not df_g.empty and not df_v.empty:
        v_minus_g_offset = float(np.median(df_v["mag"]) - np.median(df_g["mag"]))
        df_v = df_v.copy()
        df_v["mag"] = df_v["mag"].to_numpy(dtype=float) - v_minus_g_offset
        df = pd.concat([df_g, df_v], ignore_index=True)
        effective_band = "g+V_v_full_median_to_g_full_median"
    elif not df_g.empty:
        v_minus_g_offset = np.nan
        df = df_g.copy()
        effective_band = "g_only"
    elif not df_v.empty:
        v_minus_g_offset = np.nan
        df = df_v.copy()
        effective_band = "V_only_no_g_reference"
    else:
        v_minus_g_offset = np.nan
        df = pd.DataFrame(columns=_LC_COLUMNS)
        effective_band = "none"

    df = df.sort_values("JD").reset_index(drop=True)

    n_points = int(len(df))
    if n_points == 0:
        value = np.nan
        status = "no_usable_band_coverage"
    elif n_points < 2:
        value = np.nan
        status = "insufficient_points"
    else:
        value = sokolovsky_peak_to_peak_variability(
            df["mag"].to_numpy(dtype=float),
            df["error"].to_numpy(dtype=float),
        )
        status = "ok" if np.isfinite(value) else "invalid_denominator"

    return df, OrderedDict(
        [
            ("variability_sokolovsky_v", value),
            ("clipped_mean_mag_3sigma_about_median", three_sigma_clipped_mean_mag(df["mag"])),
            ("sokolovsky_v_band", effective_band),
            ("sokolovsky_v_v_minus_g_median_offset_mag", v_minus_g_offset),
            ("sokolovsky_v_n_points", n_points),
            ("sokolovsky_v_status", status),
        ]
    )


def baseline_subtracted_string_length(
    df: pd.DataFrame,
    *,
    t_col: str = "JD",
    mag_col: str = "mag",
    err_col: str = "error",
) -> dict[str, float]:
    """Compute LC string-length roughness on baseline-subtracted residuals.

    Uses per-camera baseline subtraction first, then treats the light curve as a
    single time-ordered sequence and sums |Δresidual| between consecutive points.
    Time spacing is intentionally ignored.
    """
    out: dict[str, float] = {
        "string_length_total": np.nan,
        "string_length_mean_step": np.nan,
        "string_length_n_steps": np.nan,
    }

    if not isinstance(df, pd.DataFrame) or df.empty:
        return out
    if any(col not in df.columns for col in (t_col, mag_col, err_col)):
        return out

    cols = [t_col, mag_col, err_col]
    for optional_col in ("camera_name", "camera#", "camera"):
        if optional_col in df.columns:
            cols.append(optional_col)
    work = df[cols].copy()

    work[t_col] = pd.to_numeric(work[t_col], errors="coerce")
    work[mag_col] = pd.to_numeric(work[mag_col], errors="coerce")
    work[err_col] = pd.to_numeric(work[err_col], errors="coerce")
    work = work.dropna(subset=[t_col, mag_col, err_col])
    if len(work) < 2:
        return out

    if "camera_name" in work.columns:
        cam = work["camera_name"]
    elif "camera#" in work.columns:
        cam = work["camera#"]
    elif "camera" in work.columns:
        cam = work["camera"]
    else:
        cam = pd.Series("all", index=work.index, dtype="string")

    cam = pd.Series(cam, index=work.index).astype("string").fillna("").str.strip()
    work["_stringlen_cam"] = cam.where(cam != "", "all")

    try:
        baseline_df = per_camera_gp_baseline(
            work,
            t_col=t_col,
            mag_col=mag_col,
            mag_err_col=err_col,
            cam_col="_stringlen_cam",
        )
    except Exception:
        return out

    if "resid" not in baseline_df.columns:
        return out

    t = pd.to_numeric(baseline_df[t_col], errors="coerce").to_numpy(dtype=float)
    resid = pd.to_numeric(baseline_df["resid"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(t) & np.isfinite(resid)
    if int(finite.sum()) < 2:
        return out

    order = np.argsort(t[finite], kind="mergesort")
    resid_sorted = resid[finite][order]
    step_lengths = np.abs(np.diff(resid_sorted))
    step_lengths = step_lengths[np.isfinite(step_lengths)]
    if step_lengths.size == 0:
        return out

    out["string_length_total"] = float(np.sum(step_lengths))
    out["string_length_mean_step"] = float(np.mean(step_lengths))
    out["string_length_n_steps"] = float(step_lengths.size)
    return out

def _stetson_prepare(time, mag, err):
    time = np.asarray(time, float)
    mag = np.asarray(mag, float)
    err = np.asarray(err, float)
    mask = np.isfinite(time) & np.isfinite(mag) & np.isfinite(err) & (err > 0)
    if mask.sum() < 2:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float), np.nan

    time = time[mask]
    mag = mag[mask]
    err = err[mask]
    order = np.argsort(time, kind="mergesort")
    time = time[order]
    mag = mag[order]
    err = err[order]

    mu = _stetson_robust_mean(mag, err)
    if not np.isfinite(mu):
        mu = float(np.median(mag))

    n = mag.size
    if n < 2:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float), np.nan
    delta = np.sqrt(n / (n - 1.0)) * (mag - mu) / err
    return time, mag, err, delta, float(mu)


def _stetson_groups(time, *, dtmax_days, direction, keep_singletons):
    n = len(time)
    if n == 0:
        return []

    order = list(range(n)) if direction == "forward" else list(range(n - 1, -1, -1))
    groups: list[tuple[int, ...]] = []
    pos = 0
    while pos < n:
        idx = order[pos]
        if pos + 1 < n:
            nxt = order[pos + 1]
            if abs(time[nxt] - time[idx]) <= dtmax_days:
                groups.append(tuple(sorted((idx, nxt))))
                pos += 2
                continue
        if keep_singletons:
            groups.append((idx,))
        pos += 1

    groups.sort(key=lambda group: group[0])
    return groups


def _stetson_i_from_pairs(mag, err, mu, pairs):
    n_pairs = len(pairs)
    if n_pairs < 2:
        return np.nan

    terms = [((mag[i] - mu) / err[i]) * ((mag[j] - mu) / err[j]) for i, j in pairs]
    return float(np.sqrt(1.0 / (n_pairs * (n_pairs - 1.0))) * np.sum(terms))


def _stetson_j_from_groups(delta, groups, weights=None):
    p_vals: list[float] = []
    group_weights: list[float] = []

    for group in groups:
        if len(group) == 2:
            i, j = group
            p_val = float(delta[i] * delta[j])
        elif len(group) == 1:
            (i,) = group
            p_val = float(delta[i] ** 2 - 1.0)
        else:
            continue

        weight = 1.0 if weights is None else float(weights(group))
        if not np.isfinite(weight) or weight <= 0:
            continue
        p_vals.append(p_val)
        group_weights.append(weight)

    if not p_vals:
        return np.nan

    p = np.asarray(p_vals, dtype=float)
    w = np.asarray(group_weights, dtype=float)
    return float(np.sum(w * np.sign(p) * np.sqrt(np.abs(p))) / np.sum(w))


def paper_stetson_indices(time, mag, err, *, dtmax_days=STETSON_PAIR_MAX_DT_DAYS):
    time, mag, err, delta, mu = _stetson_prepare(time, mag, err)
    if time.size < 2:
        return {
            "stetson_I": np.nan,
            "stetson_J": np.nan,
            "stetson_K": np.nan,
            "stetson_L": np.nan,
            "stetson_J_time": np.nan,
            "stetson_L_time": np.nan,
        }

    i_vals = []
    j_vals = []
    for direction in ("forward", "reverse"):
        pair_groups = [
            group
            for group in _stetson_groups(time, dtmax_days=dtmax_days, direction=direction, keep_singletons=False)
            if len(group) == 2
        ]
        i_val = _stetson_i_from_pairs(mag, err, mu, pair_groups)
        if np.isfinite(i_val):
            i_vals.append(i_val)

        groups = _stetson_groups(time, dtmax_days=dtmax_days, direction=direction, keep_singletons=True)
        j_val = _stetson_j_from_groups(delta, groups)
        if np.isfinite(j_val):
            j_vals.append(j_val)

    stetson_i = float(np.mean(i_vals)) if i_vals else np.nan
    stetson_j = float(np.mean(j_vals)) if j_vals else np.nan

    denom = np.sqrt(np.mean(np.square(delta))) if delta.size else np.nan
    stetson_k = float(np.mean(np.abs(delta)) / denom) if np.isfinite(denom) and denom > 0 else np.nan
    stetson_l = float(np.sqrt(np.pi / 2.0) * stetson_j * stetson_k) if np.isfinite(stetson_j) and np.isfinite(stetson_k) else np.nan

    dt = np.diff(time)
    if delta.size < 2 or dt.size == 0:
        stetson_j_time = np.nan
    else:
        positive_dt = dt[np.isfinite(dt) & (dt > 0)]
        median_pair_dt = float(np.median(positive_dt)) if positive_dt.size else np.nan
        if not np.isfinite(median_pair_dt) or median_pair_dt <= 0:
            weights = np.ones_like(dt, dtype=float)
        else:
            weights = np.exp(-dt / median_pair_dt)
        time_groups = [(idx, idx + 1) for idx in range(delta.size - 1)]
        stetson_j_time = _stetson_j_from_groups(
            delta,
            time_groups,
            weights=lambda group: weights[group[0]],
        )
    stetson_l_time = float(np.sqrt(np.pi / 2.0) * stetson_j_time * stetson_k) if np.isfinite(stetson_j_time) and np.isfinite(stetson_k) else np.nan

    return {
        "stetson_I": stetson_i,
        "stetson_J": stetson_j,
        "stetson_K": stetson_k,
        "stetson_L": stetson_l,
        "stetson_J_time": stetson_j_time,
        "stetson_L_time": stetson_l_time,
    }

def lomb_scargle_summary(jd, mag, err):
    if LombScargle is None:
        return {"ls_best_period_days": np.nan, "ls_peak_power": np.nan, "ls_fap": np.nan}

    t = np.asarray(jd, float)
    y = np.asarray(mag, float)
    dy = np.asarray(err, float)
    mask = np.isfinite(t) & np.isfinite(y)
    if dy.size == y.size:
        mask &= np.isfinite(dy) & (dy > 0)
    if mask.sum() < 5:
        return {"ls_best_period_days": np.nan, "ls_peak_power": np.nan, "ls_fap": np.nan}

    t = t[mask]
    y = y[mask]
    dy = dy[mask] if dy.size == mask.size else None

    if not np.isfinite(t).all():
        return {"ls_best_period_days": np.nan, "ls_peak_power": np.nan, "ls_fap": np.nan}

    if np.nanmax(t) == np.nanmin(t):
        return {"ls_best_period_days": np.nan, "ls_peak_power": np.nan, "ls_fap": np.nan}

    t = t - np.nanmin(t)
    try:
        ls = LombScargle(t, y, dy) if dy is not None else LombScargle(t, y)
        freq, power = ls.autopower()
        if power.size == 0:
            return {"ls_best_period_days": np.nan, "ls_peak_power": np.nan, "ls_fap": np.nan}
        idx = int(np.nanargmax(power))
        best_freq = float(freq[idx])
        best_period = np.nan if best_freq <= 0 or not np.isfinite(best_freq) else 1.0 / best_freq
        peak_power = float(power[idx]) if np.isfinite(power[idx]) else np.nan
        try:
            fap = float(ls.false_alarm_probability(peak_power)) if np.isfinite(peak_power) else np.nan
        except Exception:
            fap = np.nan
        return {
            "ls_best_period_days": best_period,
            "ls_peak_power": peak_power,
            "ls_fap": fap,
        }
    except Exception:
        return {"ls_best_period_days": np.nan, "ls_peak_power": np.nan, "ls_fap": np.nan}


def _block_permute_values(
    jd: np.ndarray,
    values: np.ndarray,
    rng: np.random.Generator,
    *,
    block_days: float = 1.0,
) -> np.ndarray:
    """Permute observing blocks while preserving within-block correlations."""
    jd = np.asarray(jd, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return values.copy()
    order = np.argsort(jd, kind="stable")
    times_sorted = jd[order]
    values_sorted = values[order]
    width = float(block_days)
    if not np.isfinite(width) or width <= 0:
        width = 1.0
    labels = np.floor((times_sorted - times_sorted[0]) / width).astype(np.int64)
    blocks = [values_sorted[labels == label] for label in np.unique(labels)]
    if len(blocks) <= 1:
        shift = int(rng.integers(1, values_sorted.size))
        permuted_sorted = np.roll(values_sorted, shift)
    else:
        block_order = rng.permutation(len(blocks))
        permuted_sorted = np.concatenate([blocks[int(idx)] for idx in block_order])
    # Blocks can have unequal lengths; assigning their concatenation to the
    # fixed time grid preserves local correlation without preserving phase.
    out = np.empty_like(values_sorted)
    out[:] = permuted_sorted
    restored = np.empty_like(out)
    restored[order] = out
    return restored


def bootstrap_lomb_scargle(
    jd: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    min_frequency: float = LS_MIN_FREQUENCY,
    max_frequency: float = LS_MAX_FREQUENCY,
    exclude_alias_periods: bool = True,
    alias_tolerance: float = LS_ALIAS_TOLERANCE,
    significance_level: float = 0.01,
    random_state: int | None = 0,
    block_days: float = 1.0,
) -> dict:
    """
    Bootstrap Lomb-Scargle periodogram with significance testing.

    More robust than simple FAP - uses bootstrap shuffling to determine
    empirical significance of the peak power.

    Parameters
    ----------
    jd : array
        Julian dates
    mag : array
        Magnitudes
    err : array
        Magnitude errors
    n_bootstrap : int
        Number of bootstrap iterations (default 1000)
    min_frequency : float
        Minimum frequency to search (default 1/365.25 = 1 year period)
    max_frequency : float
        Maximum frequency to search (default 10 = 0.1 day period)
    exclude_alias_periods : bool
        Flag if best period is near known aliases
    alias_tolerance : float
        Tolerance for alias matching in days (default 0.1)

    Returns
    -------
    dict with keys:
        - ls_power: float, peak power
        - ls_period_days: float, best period
        - ls_bootstrap_sig: float, bootstrap significance (fraction of shuffles with higher power)
        - ls_is_alias: bool, True if near known alias period
        - ls_is_significant: bool, True if bootstrap_sig < 0.01 and not alias
    """
    empty = {
            "ls_power": np.nan,
            "ls_period_days": np.nan,
            "ls_bootstrap_sig": np.nan,
            "ls_is_alias": False,
            "ls_is_significant": False,
            "ls_bootstrap_attempted": int(max(n_bootstrap, 0)),
            "ls_bootstrap_successful": 0,
            "ls_bootstrap_method": "observing_block_permutation",
            "ls_status": "unavailable",
        }
    if LombScargle is None:
        return empty

    jd = np.asarray(jd, float)
    mag = np.asarray(mag, float)
    err = np.asarray(err, float)

    mask = np.isfinite(jd) & np.isfinite(mag) & np.isfinite(err) & (err > 0)
    if mask.sum() < 50:
        return {**empty, "ls_status": "insufficient_points"}

    jd = jd[mask]
    mag = mag[mask]
    err = err[mask]

    order = np.argsort(jd, kind="stable")
    jd, mag, err = jd[order], mag[order], err[order]

    try:
        ls = LombScargle(jd, mag, err)
        freq, power_spec = ls.autopower(
            minimum_frequency=min_frequency,
            maximum_frequency=max_frequency,
        )
        if power_spec.size == 0 or not np.isfinite(power_spec).any():
            return {**empty, "ls_status": "empty_periodogram"}

        max_idx = int(np.nanargmax(power_spec))
        ls_power = float(power_spec[max_idx])
        best_period = float(1.0 / freq[max_idx]) if freq[max_idx] > 0 else np.nan

        rng = np.random.default_rng(random_state)
        bootstrap_powers = np.full(int(max(n_bootstrap, 0)), np.nan, dtype=float)
        for idx in range(bootstrap_powers.size):
            shuffled_mag = _block_permute_values(jd, mag, rng, block_days=block_days)
            try:
                power_boot = LombScargle(jd, shuffled_mag, err).power(freq)
                if power_boot.size and np.isfinite(power_boot).any():
                    bootstrap_powers[idx] = float(np.nanmax(power_boot))
            except Exception:
                continue
        finite = bootstrap_powers[np.isfinite(bootstrap_powers)]
        bootstrap_sig = (
            float((np.count_nonzero(finite >= ls_power) + 1) / (finite.size + 1))
            if finite.size else np.nan
        )
        span = float(np.ptp(jd)) if jd.size > 1 else np.nan
        aliases = period_alias_matches(
            best_period,
            alias_periods=LS_ALIAS_PERIODS,
            alias_tolerance=alias_tolerance,
            time_span_days=span,
        ) if exclude_alias_periods else []
        is_alias = bool(aliases)
        is_significant = bool(
            np.isfinite(bootstrap_sig)
            and bootstrap_sig < float(significance_level)
            and not is_alias
        )
        return {
            "ls_power": ls_power,
            "ls_period_days": best_period,
            "ls_bootstrap_sig": bootstrap_sig,
            "ls_is_alias": is_alias,
            "ls_alias_matches": aliases,
            "ls_is_significant": is_significant,
            "ls_bootstrap_attempted": int(bootstrap_powers.size),
            "ls_bootstrap_successful": int(finite.size),
            "ls_bootstrap_method": "observing_block_permutation",
            "ls_status": "ok" if finite.size or bootstrap_powers.size == 0 else "bootstrap_failed",
        }
    except Exception:
        return {**empty, "ls_status": "error"}


def long_period_ls_search(
    jd: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    *,
    stage: str = "long",
    baseline_days: float | None = None,
    cadence_median_days: float | None = None,
    n_bootstrap: int = 200,
    samples_per_peak: int = 10,
    block_days: float = 30.0,
    significance_level: float = 0.01,
    random_state: int | None = 0,
    min_period_days: float | None = None,
    max_period_days: float | None = None,
) -> dict:
    """Search for long-period signals with a baseline-adaptive frequency range.

    This complements ``bootstrap_lomb_scargle`` which is bounded at
    ``LS_MIN_FREQUENCY = 1/365.25``. Long-recurrence variables (e.g. AA-Tau-like
    dippers with two dips separated by ~2000 d) live below that frequency and
    need a search window scaled to the observed baseline.

    Bootstrap FAP is estimated with block permutation using a 30-day block by
    default so short-period correlations do not leak into long-period power.

    Parameters
    ----------
    jd, mag, err : array-like
        Julian dates, magnitudes, magnitude errors. Non-finite / non-positive
        error entries are dropped.
    stage : str
        Bounds stage forwarded to ``adaptive_period_bounds`` when
        ``min_period_days`` / ``max_period_days`` are not supplied. Defaults
        to ``"long"`` (see ``malca.core.period_bounds``).
    baseline_days, cadence_median_days :
        If not provided they are derived from ``jd``.
    n_bootstrap :
        Number of block-permutation bootstrap iterations. 200 is a reasonable
        default; use >=500 when you need tight FAPs.
    samples_per_peak :
        LombScargle oversampling factor. Higher = finer period resolution at
        the cost of runtime; 10 is a safe default for long-P searches.
    block_days :
        Permutation block width in days. Larger blocks preserve long-timescale
        correlated noise so we do not spuriously reject red-noise periods.
    significance_level :
        FAP threshold to set ``long_ls_is_significant``.
    random_state :
        Seed for reproducibility.

    Returns
    -------
    dict with keys:
        long_ls_period_days, long_ls_peak_power,
        long_ls_fap_bootstrap, long_ls_baseline_cycles,
        long_ls_is_significant, long_ls_min_period_days,
        long_ls_max_period_days, long_ls_n_bootstrap_attempted,
        long_ls_n_bootstrap_successful, long_ls_status,
        long_ls_alias_matches, long_ls_is_alias.
    """
    from malca.core.period_bounds import adaptive_period_bounds

    empty = {
        "long_ls_period_days": np.nan,
        "long_ls_peak_power": np.nan,
        "long_ls_fap_bootstrap": np.nan,
        "long_ls_baseline_cycles": np.nan,
        "long_ls_is_significant": False,
        "long_ls_min_period_days": np.nan,
        "long_ls_max_period_days": np.nan,
        "long_ls_n_bootstrap_attempted": int(max(n_bootstrap, 0)),
        "long_ls_n_bootstrap_successful": 0,
        "long_ls_status": "unavailable",
        "long_ls_alias_matches": [],
        "long_ls_is_alias": False,
        "long_ls_top_periods_days": [],
        "long_ls_top_powers": [],
    }

    if LombScargle is None:
        return empty

    jd = np.asarray(jd, dtype=float)
    mag = np.asarray(mag, dtype=float)
    err = np.asarray(err, dtype=float)
    mask = np.isfinite(jd) & np.isfinite(mag)
    if err.size == mag.size:
        mask &= np.isfinite(err) & (err > 0)
    if mask.sum() < 20:
        return {**empty, "long_ls_status": "insufficient_points"}

    jd = jd[mask]
    mag = mag[mask]
    err = err[mask] if err.size == mask.size else None

    order = np.argsort(jd, kind="stable")
    jd = jd[order]
    mag = mag[order]
    if err is not None:
        err = err[order]

    baseline = (
        float(baseline_days)
        if baseline_days is not None and np.isfinite(baseline_days) and float(baseline_days) > 0
        else float(jd[-1] - jd[0])
    )
    if not np.isfinite(baseline) or baseline <= 0:
        return {**empty, "long_ls_status": "zero_baseline"}

    if cadence_median_days is None:
        diffs = np.diff(jd)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        cadence_median_days = float(np.median(diffs)) if diffs.size else None

    bounds = adaptive_period_bounds(
        baseline_days=baseline,
        stage=stage,
        cadence_median_days=cadence_median_days,
        n_points=int(jd.size),
        user_min_period=min_period_days,
        user_max_period=max_period_days,
    )
    min_period, max_period = bounds.as_tuple()
    if max_period <= min_period:
        return {**empty, "long_ls_status": "invalid_bounds"}

    min_freq = 1.0 / max_period
    max_freq = 1.0 / min_period

    try:
        ls = LombScargle(jd, mag, err) if err is not None else LombScargle(jd, mag)
        freq, power_spec = ls.autopower(
            minimum_frequency=min_freq,
            maximum_frequency=max_freq,
            samples_per_peak=samples_per_peak,
        )
        if power_spec.size == 0 or not np.isfinite(power_spec).any():
            return {**empty, "long_ls_status": "empty_periodogram"}

        max_idx = int(np.nanargmax(power_spec))
        peak_power = float(power_spec[max_idx])
        best_freq = float(freq[max_idx])
        best_period = float(1.0 / best_freq) if best_freq > 0 else np.nan
        baseline_cycles = float(baseline / best_period) if best_period > 0 else np.nan

        top_periods, top_powers = _select_long_ls_top_peaks(
            freq=np.asarray(freq),
            power=np.asarray(power_spec),
            top_k=5,
            min_period=min_period,
            max_period=max_period,
        )

        rng = np.random.default_rng(random_state)
        n_boot = int(max(n_bootstrap, 0))
        boot_powers = np.full(n_boot, np.nan, dtype=float)
        for idx in range(n_boot):
            shuffled = _block_permute_values(jd, mag, rng, block_days=block_days)
            try:
                power_boot = LombScargle(jd, shuffled, err).power(freq) if err is not None else LombScargle(jd, shuffled).power(freq)
                if power_boot.size and np.isfinite(power_boot).any():
                    boot_powers[idx] = float(np.nanmax(power_boot))
            except Exception:
                continue

        finite = boot_powers[np.isfinite(boot_powers)]
        fap = (
            float((np.count_nonzero(finite >= peak_power) + 1) / (finite.size + 1))
            if finite.size
            else np.nan
        )
        span = float(np.ptp(jd)) if jd.size > 1 else np.nan
        aliases = period_alias_matches(
            best_period,
            alias_periods=LS_ALIAS_PERIODS,
            alias_tolerance=LS_ALIAS_TOLERANCE,
            time_span_days=span,
        )
        is_alias = bool(aliases)
        is_significant = bool(
            np.isfinite(fap)
            and fap < float(significance_level)
            and not is_alias
        )
        return {
            "long_ls_period_days": best_period,
            "long_ls_peak_power": peak_power,
            "long_ls_fap_bootstrap": fap,
            "long_ls_baseline_cycles": baseline_cycles,
            "long_ls_is_significant": is_significant,
            "long_ls_min_period_days": float(min_period),
            "long_ls_max_period_days": float(max_period),
            "long_ls_n_bootstrap_attempted": int(n_boot),
            "long_ls_n_bootstrap_successful": int(finite.size),
            "long_ls_status": "ok" if finite.size or n_boot == 0 else "bootstrap_failed",
            "long_ls_alias_matches": aliases,
            "long_ls_is_alias": is_alias,
            "long_ls_top_periods_days": top_periods,
            "long_ls_top_powers": top_powers,
        }
    except Exception as exc:
        return {**empty, "long_ls_status": f"error:{type(exc).__name__}"}


def _select_long_ls_top_peaks(
    *,
    freq: np.ndarray,
    power: np.ndarray,
    top_k: int,
    min_period: float,
    max_period: float,
) -> tuple[list[float], list[float]]:
    """Return the top-K well-separated periodogram peaks.

    Peaks are enforced to be at least ~2 frequency bins apart so a single broad
    peak does not consume the whole shortlist. The output is sorted by
    descending power.
    """
    freq = np.asarray(freq, dtype=float)
    power = np.asarray(power, dtype=float)
    if freq.size == 0 or power.size == 0:
        return [], []
    finite = np.isfinite(power) & np.isfinite(freq) & (freq > 0)
    if not finite.any():
        return [], []

    periods = np.where(finite, 1.0 / np.where(freq == 0, np.nan, freq), np.nan)
    valid = finite & np.isfinite(periods) & (periods >= float(min_period)) & (periods <= float(max_period))
    if not valid.any():
        return [], []

    order = np.argsort(-power)
    order = order[valid[order]]

    chosen_idx: list[int] = []
    for idx in order:
        i = int(idx)
        if any(abs(i - j) < 2 for j in chosen_idx):
            continue
        chosen_idx.append(i)
        if len(chosen_idx) >= int(top_k):
            break

    return (
        [float(periods[i]) for i in chosen_idx],
        [float(power[i]) for i in chosen_idx],
    )


def _bootstrap_min_metric(
    period_finder,
    jd: np.ndarray,
    mag: np.ndarray,
    *,
    n_bootstrap: int,
    min_period: float,
    max_period: float,
    n_periods: int,
    period_finder_kwargs: dict | None = None,
    random_state: int | None = 0,
    block_days: float = 1.0,
) -> np.ndarray:
    """Return bootstrap distribution of minimum periodogram statistic."""
    n_bootstrap = int(max(n_bootstrap, 0))
    if n_bootstrap == 0:
        return np.empty(0, dtype=float)

    period_finder_kwargs = dict(period_finder_kwargs or {})
    mins = np.full(n_bootstrap, np.nan, dtype=float)
    rng = np.random.default_rng(random_state)
    for i in range(n_bootstrap):
        try:
            shuffled_mag = _block_permute_values(jd, mag, rng, block_days=block_days)
            _, _, metric = period_finder(
                jd,
                shuffled_mag,
                min_period=min_period,
                max_period=max_period,
                n_periods=n_periods,
                **period_finder_kwargs,
            )
            if metric.size > 0:
                mins[i] = float(np.min(metric))
        except Exception:
            mins[i] = np.nan

    return mins


def compute_pdm_stats(
    jd: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    min_period: float = 1.0,
    max_period: float = 100.0,
    n_periods: int = 10000,
    pdm_method: str = "classic",
    pdm_phase_width: float = PDM_PLAVCHAN_PHASE_WIDTH,
    pdm_min_neighbors: int = PDM_PLAVCHAN_MIN_NEIGHBORS,
    n_bootstrap: int = 0,
    significance_level: float = 0.01,
    refine: bool = False,
    refine_top_k: int = PERIODOGRAM_REFINE_TOP_K,
    refine_window_steps: float = PERIODOGRAM_REFINE_WINDOW_STEPS,
    refine_n_grid: int = PERIODOGRAM_REFINE_N_GRID,
    random_state: int | None = 0,
    bootstrap_block_days: float = 1.0,
) -> dict:
    """
    Run Phase Dispersion Minimization and compute significance metrics.

    Returns:
        pdm_period: float, best period
        pdm_min_theta: float, lowest PDM score (classical theta for classic PDM;
            worst-k normalized residual score for the Plavchan variant)
        pdm_snr: float, SNR of the theta dip relative to background
        pdm_bootstrap_sig: float, bootstrap significance (fraction of shuffles with lower/equal theta)
        pdm_is_significant: bool, True if bootstrap significance is below threshold
    """
    out = {
        "pdm_period": np.nan,
        "pdm_min_theta": np.nan,
        "pdm_snr": np.nan,
        "pdm_bootstrap_sig": np.nan,
        "pdm_is_significant": False,
        "pdm_bootstrap_attempted": int(max(n_bootstrap, 0)),
        "pdm_bootstrap_successful": 0,
        "pdm_bootstrap_method": "observing_block_permutation",
        "pdm_status": "insufficient_points",
    }

    mask = np.isfinite(jd) & np.isfinite(mag)
    if mask.sum() < 50:
        return out

    try:
        jd_clean = np.asarray(jd[mask], dtype=float)
        mag_clean = np.asarray(mag[mask], dtype=float)

        best_p, _, thetas = pdm_find_period(
            jd_clean,
            mag_clean,
            min_period=min_period,
            max_period=max_period,
            n_periods=n_periods,
            method=pdm_method,
            phase_width=pdm_phase_width,
            min_neighbors=pdm_min_neighbors,
            refine=bool(refine),
            refine_top_k=int(refine_top_k),
            refine_window_steps=float(refine_window_steps),
            refine_n_grid=int(refine_n_grid),
        )

        min_theta = float(np.min(thetas))
        theta_std = float(np.std(thetas))
        if np.isfinite(theta_std) and theta_std > 0:
            pdm_snr = float((np.mean(thetas) - min_theta) / theta_std)
        else:
            pdm_snr = np.nan

        out.update({
            "pdm_period": float(best_p),
            "pdm_min_theta": min_theta,
            "pdm_snr": pdm_snr,
            "pdm_status": "ok",
        })

        if int(n_bootstrap) > 0 and np.isfinite(min_theta):
            null_min_theta = _bootstrap_min_metric(
                pdm_find_period,
                jd_clean,
                mag_clean,
                n_bootstrap=n_bootstrap,
                min_period=min_period,
                max_period=max_period,
                n_periods=n_periods,
                period_finder_kwargs={
                    "method": pdm_method,
                    "phase_width": pdm_phase_width,
                    "min_neighbors": pdm_min_neighbors,
                },
                random_state=random_state,
                block_days=bootstrap_block_days,
            )
            finite = null_min_theta[np.isfinite(null_min_theta)]
            out["pdm_bootstrap_successful"] = int(finite.size)
            if finite.size > 0:
                bootstrap_sig = float(
                    (np.count_nonzero(finite <= min_theta) + 1) / (finite.size + 1)
                )
                out["pdm_bootstrap_sig"] = bootstrap_sig
                out["pdm_is_significant"] = bool(bootstrap_sig < float(significance_level))
            else:
                out["pdm_status"] = "bootstrap_failed"

        return out
    except Exception:
        out["pdm_status"] = "error"
        return out

def compute_ce_stats(
    jd: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    min_period: float = 1.0,
    max_period: float = 100.0,
    n_periods: int = 10000,
    n_bootstrap: int = 0,
    significance_level: float = 0.01,
    refine: bool = False,
    refine_top_k: int = PERIODOGRAM_REFINE_TOP_K,
    refine_window_steps: float = PERIODOGRAM_REFINE_WINDOW_STEPS,
    refine_n_grid: int = PERIODOGRAM_REFINE_N_GRID,
    random_state: int | None = 0,
    bootstrap_block_days: float = 1.0,
) -> dict:
    """
    Run Conditional Entropy and compute significance metrics.

    Returns:
        ce_period: float, best period
        ce_min_entropy: float, lowest entropy value
        ce_snr: float, SNR of the entropy dip relative to background
        ce_bootstrap_sig: float, bootstrap significance (fraction of shuffles with lower/equal entropy)
        ce_is_significant: bool, True if bootstrap significance is below threshold
    """
    out = {
        "ce_period": np.nan,
        "ce_min_entropy": np.nan,
        "ce_snr": np.nan,
        "ce_bootstrap_sig": np.nan,
        "ce_is_significant": False,
        "ce_bootstrap_attempted": int(max(n_bootstrap, 0)),
        "ce_bootstrap_successful": 0,
        "ce_bootstrap_method": "observing_block_permutation",
        "ce_status": "insufficient_points",
    }

    mask = np.isfinite(jd) & np.isfinite(mag)
    if mask.sum() < 50:
        return out

    try:
        jd_clean = np.asarray(jd[mask], dtype=float)
        mag_clean = np.asarray(mag[mask], dtype=float)

        best_p, _, entropies = ce_find_period(
            jd_clean,
            mag_clean,
            min_period=min_period,
            max_period=max_period,
            n_periods=n_periods,
            refine=bool(refine),
            refine_top_k=int(refine_top_k),
            refine_window_steps=float(refine_window_steps),
            refine_n_grid=int(refine_n_grid),
        )
        min_entropy = float(np.min(entropies))
        entropy_std = float(np.std(entropies))
        if np.isfinite(entropy_std) and entropy_std > 0:
            ce_snr = float((np.mean(entropies) - min_entropy) / entropy_std)
        else:
            ce_snr = np.nan

        out.update({
            "ce_period": float(best_p),
            "ce_min_entropy": min_entropy,
            "ce_snr": ce_snr,
            "ce_status": "ok",
        })

        if int(n_bootstrap) > 0 and np.isfinite(min_entropy):
            null_min_entropy = _bootstrap_min_metric(
                ce_find_period,
                jd_clean,
                mag_clean,
                n_bootstrap=n_bootstrap,
                min_period=min_period,
                max_period=max_period,
                n_periods=n_periods,
                random_state=random_state,
                block_days=bootstrap_block_days,
            )
            finite = null_min_entropy[np.isfinite(null_min_entropy)]
            out["ce_bootstrap_successful"] = int(finite.size)
            if finite.size > 0:
                bootstrap_sig = float(
                    (np.count_nonzero(finite <= min_entropy) + 1) / (finite.size + 1)
                )
                out["ce_bootstrap_sig"] = bootstrap_sig
                out["ce_is_significant"] = bool(bootstrap_sig < float(significance_level))
            else:
                out["ce_status"] = "bootstrap_failed"

        return out
    except Exception:
        out["ce_status"] = "error"
        return out

def linear_trend(x, y):
    # returns slope, intercept, r^2; robust to NaNs
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2: return np.nan, np.nan, np.nan
    x = x[mask]; y = y[mask]
    p = np.polyfit(x, y, 1)
    yhat = np.polyval(p, x)
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return float(p[0]), float(p[1]), float(r2)

# ---------------------------------------------------------------------------
# ALeRCE-style light-curve features
# ---------------------------------------------------------------------------

def amplitude(mag):
    """Half the difference between median of top 5% and bottom 5% magnitudes."""
    mag = np.asarray(mag, float)
    mag = mag[np.isfinite(mag)]
    if mag.size < 20:
        return np.nan
    n5 = max(1, int(np.ceil(0.05 * mag.size)))
    sorted_mag = np.sort(mag)
    return float((np.median(sorted_mag[-n5:]) - np.median(sorted_mag[:n5])) / 2.0)


def beyond_1_std(mag):
    """Fraction of points beyond 1 sigma from the weighted mean."""
    mag = np.asarray(mag, float)
    mag = mag[np.isfinite(mag)]
    if mag.size < 3:
        return np.nan
    mu = np.mean(mag)
    sigma = np.std(mag, ddof=1)
    if sigma <= 0:
        return np.nan
    return float(np.sum(np.abs(mag - mu) > sigma) / mag.size)


def con(mag, threshold=2.0):
    """Number of 3 consecutive points brighter/fainter than threshold*sigma."""
    mag = np.asarray(mag, float)
    mag = mag[np.isfinite(mag)]
    if mag.size < 3:
        return 0
    mu = np.mean(mag)
    sigma = np.std(mag, ddof=1)
    if sigma <= 0:
        return 0
    beyond = np.abs(mag - mu) > threshold * sigma
    count = 0
    for i in range(len(beyond) - 2):
        if beyond[i] and beyond[i + 1] and beyond[i + 2]:
            count += 1
    return count


def delta_mag(mag):
    """Difference between max and min magnitude."""
    mag = np.asarray(mag, float)
    mag = mag[np.isfinite(mag)]
    if mag.size < 2:
        return np.nan
    return float(np.max(mag) - np.min(mag))


def intrinsic_sigma_mag(mag, err):
    """Magnitude-space intrinsic scatter: sqrt(max(Var - mean(err^2), 0))."""
    mag = np.asarray(mag, float)
    err = np.asarray(err, float)
    mask = np.isfinite(mag) & np.isfinite(err) & (err > 0)
    if mask.sum() < 3:
        return np.nan
    mag = mag[mask]
    err = err[mask]
    var = np.var(mag, ddof=1)
    mean_err2 = np.mean(err ** 2)
    inner = var - mean_err2
    return float(np.sqrt(inner)) if inner > 0 else 0.0


def gskew(mag):
    """Median-based skewness: (mean - median) / std."""
    mag = np.asarray(mag, float)
    mag = mag[np.isfinite(mag)]
    if mag.size < 3:
        return np.nan
    sigma = np.std(mag, ddof=1)
    if sigma <= 0:
        return np.nan
    return float((np.mean(mag) - np.median(mag)) / sigma)


def max_slope(mag, time):
    """Maximum absolute magnitude slope between consecutive observations."""
    mag = np.asarray(mag, float)
    time = np.asarray(time, float)
    mask = np.isfinite(mag) & np.isfinite(time)
    if mask.sum() < 2:
        return np.nan
    mag = mag[mask]
    time = time[mask]
    dt = np.diff(time)
    dm = np.abs(np.diff(mag))
    valid = dt > 0
    if not np.any(valid):
        return np.nan
    slopes = dm[valid] / dt[valid]
    return float(np.max(slopes))


def meanvariance(mag):
    """Ratio of standard deviation to mean magnitude."""
    mag = np.asarray(mag, float)
    mag = mag[np.isfinite(mag)]
    if mag.size < 3:
        return np.nan
    mu = np.mean(mag)
    if mu == 0:
        return np.nan
    return float(np.std(mag, ddof=1) / mu)


def median_abs_dev(mag):
    """Median absolute deviation (raw, not scaled)."""
    mag = np.asarray(mag, float)
    mag = mag[np.isfinite(mag)]
    if mag.size == 0:
        return np.nan
    return float(np.median(np.abs(mag - np.median(mag))))


def median_brp(mag):
    """Fraction of points within amplitude/10 of the median."""
    mag = np.asarray(mag, float)
    mag = mag[np.isfinite(mag)]
    if mag.size < 5:
        return np.nan
    amp = amplitude(mag)
    if not np.isfinite(amp) or amp <= 0:
        return np.nan
    med = np.median(mag)
    return float(np.sum(np.abs(mag - med) < amp / 10.0) / mag.size)


def percent_amplitude(mag):
    """Largest percentage difference between max or min and median."""
    mag = np.asarray(mag, float)
    mag = mag[np.isfinite(mag)]
    if mag.size < 3:
        return np.nan
    med = np.median(mag)
    if med == 0:
        return np.nan
    return float(max(abs(np.max(mag) - med), abs(np.min(mag) - med)) / abs(med))


def ahl_ratio(mag):
    """Ratio of points brighter than the mean to points fainter than the mean.

    This follows the ASAS-SN ``AHL`` feature convention in magnitude space:
    brighter points have smaller magnitudes than the mean.
    """
    mag = np.asarray(mag, float)
    mag = mag[np.isfinite(mag)]
    if mag.size < 3:
        return np.nan
    mean_mag = float(np.mean(mag))
    n_brighter = int(np.sum(mag < mean_mag))
    n_fainter = int(np.sum(mag > mean_mag))
    if n_fainter <= 0:
        return np.nan
    return float(n_brighter / n_fainter)


def q31(mag):
    """Difference between 75th and 25th percentile."""
    mag = np.asarray(mag, float)
    mag = mag[np.isfinite(mag)]
    if mag.size < 4:
        return np.nan
    return float(np.percentile(mag, 75) - np.percentile(mag, 25))


def skew(mag):
    """Skewness of the magnitude distribution."""
    mag = np.asarray(mag, float)
    mag = mag[np.isfinite(mag)]
    if mag.size < 3:
        return np.nan
    return float(sp_stats.skew(mag, bias=False))


def small_kurtosis(mag):
    """Small-sample kurtosis of magnitudes."""
    mag = np.asarray(mag, float)
    mag = mag[np.isfinite(mag)]
    if mag.size < 4:
        return np.nan
    return float(sp_stats.kurtosis(mag, bias=False))


def constancy_p_value(mag, err):
    """Chi-square tail p-value under a constant-flux null model."""
    mag = np.asarray(mag, float)
    err = np.asarray(err, float)
    mask = np.isfinite(mag) & np.isfinite(err) & (err > 0)
    if mask.sum() < 3:
        return np.nan
    mag = mag[mask]
    err = err[mask]
    w = 1.0 / err ** 2
    mu = np.sum(w * mag) / np.sum(w)
    chi2_val = np.sum(((mag - mu) / err) ** 2)
    dof = len(mag) - 1
    if dof <= 0:
        return np.nan
    return float(1.0 - sp_stats.chi2.cdf(chi2_val, dof))


def anderson_darling(mag):
    """Anderson-Darling test statistic for normality."""
    mag = np.asarray(mag, float)
    mag = mag[np.isfinite(mag)]
    if mag.size < 8 or np.std(mag) == 0:
        return np.nan
    result = sp_stats.anderson(mag, dist='norm')
    return float(result.statistic)


def pair_slope_trend(mag, n=30):
    """Fraction of increasing minus decreasing consecutive differences (last n points)."""
    mag = np.asarray(mag, float)
    mag = mag[np.isfinite(mag)]
    if mag.size < 3:
        return np.nan
    tail = mag[-n:] if mag.size >= n else mag
    diffs = np.diff(tail)
    if diffs.size == 0:
        return np.nan
    n_inc = np.sum(diffs > 0)
    n_dec = np.sum(diffs < 0)
    return float((n_inc - n_dec) / diffs.size)


def rcs(mag):
    """Range of cumulative sum."""
    mag = np.asarray(mag, float)
    mag = mag[np.isfinite(mag)]
    n = mag.size
    if n < 3:
        return np.nan
    sigma = np.std(mag, ddof=1)
    if sigma <= 0:
        return np.nan
    s = np.cumsum(mag - np.mean(mag)) / (n * sigma)
    return float(np.max(s) - np.min(s))


def autocor_length(mag, eta_e):
    """Lag where ACF drops below eta_e (von Neumann ratio)."""
    mag = np.asarray(mag, float)
    mag = mag[np.isfinite(mag)]
    n = mag.size
    if n < 10 or not np.isfinite(eta_e):
        return 0
    mu = np.mean(mag)
    var = np.var(mag, ddof=0)
    if var <= 0:
        return 0
    max_lag = min(n // 2, 100)
    for lag in range(1, max_lag + 1):
        acf = np.mean((mag[:n - lag] - mu) * (mag[lag:] - mu)) / var
        if acf < eta_e:
            return lag
    return max_lag


def structure_function(mag, time):
    """
    Structure function analysis.

    Returns (sf_amplitude, sf_gamma) where:
    - sf_amplitude = sqrt(SF at 1 year timescale)
    - sf_gamma = log slope of SF vs tau
    """
    mag = np.asarray(mag, float)
    time = np.asarray(time, float)
    mask = np.isfinite(mag) & np.isfinite(time)
    if mask.sum() < 10:
        return np.nan, np.nan
    mag = mag[mask]
    time = time[mask]

    # compute all pairwise differences (upper triangle only)
    n = len(mag)
    dt_list = []
    dm2_list = []
    for i in range(n):
        dt_arr = time[i + 1:] - time[i]
        dm_arr = (mag[i + 1:] - mag[i]) ** 2
        dt_list.append(dt_arr)
        dm2_list.append(dm_arr)

    all_dt = np.concatenate(dt_list)
    all_dm2 = np.concatenate(dm2_list)
    valid = all_dt > 0
    all_dt = all_dt[valid]
    all_dm2 = all_dm2[valid]

    if len(all_dt) < 10:
        return np.nan, np.nan

    # bin by log(dt)
    log_dt = np.log10(all_dt)
    n_bins = 20
    bins = np.linspace(log_dt.min(), log_dt.max(), n_bins + 1)
    bin_centers = []
    bin_sf = []
    for i in range(n_bins):
        in_bin = (log_dt >= bins[i]) & (log_dt < bins[i + 1])
        if in_bin.sum() >= 3:
            bin_centers.append((bins[i] + bins[i + 1]) / 2.0)
            bin_sf.append(np.mean(all_dm2[in_bin]))

    if len(bin_centers) < 3:
        return np.nan, np.nan

    log_tau = np.array(bin_centers)
    log_sf = np.log10(np.maximum(np.array(bin_sf), 1e-10))
    valid_sf = np.isfinite(log_sf)
    if valid_sf.sum() < 3:
        return np.nan, np.nan

    # linear fit in log-log space: log(SF) = gamma * log(tau) + const
    coeffs = np.polyfit(log_tau[valid_sf], log_sf[valid_sf], 1)
    sf_gamma = float(coeffs[0])

    # SF at 1 year (365.25 days)
    log_tau_1yr = np.log10(365.25)
    sf_at_1yr = 10.0 ** (coeffs[0] * log_tau_1yr + coeffs[1])
    sf_amplitude = float(np.sqrt(sf_at_1yr))

    return sf_amplitude, sf_gamma


def lafler_kinman_t(mag, order=None, *, wrap: bool = True) -> float:
    """Normalized Lafler-Kinman successive-difference statistic."""
    mag = np.asarray(mag, float)
    if order is not None:
        order = np.asarray(order)
        if order.size != mag.size:
            return np.nan
        mag = mag[np.argsort(order, kind="mergesort")]
    mag = mag[np.isfinite(mag)]
    if mag.size < 3:
        return np.nan

    diffs = np.diff(mag)
    if wrap:
        diffs = np.concatenate((diffs, np.asarray([mag[0] - mag[-1]], dtype=float)))
    denom = float(np.sum(np.square(mag - np.mean(mag))))
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    return float(np.sum(np.square(diffs)) / denom)


def lafler_kinman_period_stats(time, mag, period) -> dict[str, float]:
    """Return ASAS-SN-style time and folded Lafler-Kinman metrics."""
    out = {
        "lafler_kinman_t_time": np.nan,
        "lafler_kinman_t_phase": np.nan,
        "lafler_kinman_delta": np.nan,
    }
    time = np.asarray(time, float)
    mag = np.asarray(mag, float)
    mask = np.isfinite(time) & np.isfinite(mag)
    if int(mask.sum()) < 3:
        return out
    time = time[mask]
    mag = mag[mask]

    t_time = lafler_kinman_t(mag, order=time, wrap=False)
    out["lafler_kinman_t_time"] = t_time

    if np.isfinite(period) and float(period) > 0:
        phase = np.mod((time - np.nanmin(time)) / float(period), 1.0)
        t_phase = lafler_kinman_t(mag, order=phase, wrap=True)
        out["lafler_kinman_t_phase"] = t_phase
        if np.isfinite(t_phase) and np.isfinite(t_time) and t_time != 0:
            out["lafler_kinman_delta"] = float((t_phase - t_time) / t_time)
    return out


def window_function_alias_peaks(
    time,
    *,
    min_frequency: float = LS_MIN_FREQUENCY,
    max_frequency: float = LS_MAX_FREQUENCY,
    n_peaks: int = 5,
    n_frequency: int = 2048,
) -> dict[str, float]:
    """Return strongest sampling-window periods and powers.

    The spectral window power is ``|sum(exp(2*pi*i*f*t))|^2 / N^2``.
    """
    out: dict[str, float] = {}
    for idx in range(1, int(n_peaks) + 1):
        out[f"window_alias_period_{idx}"] = np.nan
        out[f"window_alias_power_{idx}"] = np.nan

    time = np.asarray(time, float)
    time = time[np.isfinite(time)]
    if time.size < 3:
        return out
    min_frequency = float(min_frequency)
    max_frequency = float(max_frequency)
    if not np.isfinite(min_frequency) or not np.isfinite(max_frequency) or max_frequency <= min_frequency:
        return out

    centered = time - np.min(time)
    frequency = np.linspace(min_frequency, max_frequency, max(32, int(n_frequency)))
    power = np.empty(frequency.size, dtype=float)
    chunk_size = 512
    norm = float(time.size * time.size)
    for start in range(0, frequency.size, chunk_size):
        stop = min(start + chunk_size, frequency.size)
        phase = 2.0 * np.pi * frequency[start:stop, None] * centered[None, :]
        power[start:stop] = np.square(np.abs(np.sum(np.exp(1j * phase), axis=1))) / norm
    if power.size == 0 or not np.isfinite(power).any():
        return out

    chosen: list[int] = []
    min_sep = max(1, int(round(power.size / max(50, 10 * int(n_peaks)))))
    for idx in np.argsort(-power):
        idx = int(idx)
        if not np.isfinite(power[idx]):
            continue
        if any(abs(idx - prev) < min_sep for prev in chosen):
            continue
        chosen.append(idx)
        if len(chosen) >= int(n_peaks):
            break

    for rank, idx in enumerate(chosen, start=1):
        freq = float(frequency[idx])
        out[f"window_alias_period_{rank}"] = float(1.0 / freq) if freq > 0 else np.nan
        out[f"window_alias_power_{rank}"] = float(power[idx])
    return out


def eb_minima_ratio(mag, time, period) -> dict[str, float]:
    """Estimate the primary/secondary eclipse depth ratio from a folded curve."""
    out = {
        "eb_rminima": np.nan,
        "eb_primary_min_depth": np.nan,
        "eb_secondary_min_depth": np.nan,
    }
    mag = np.asarray(mag, float)
    time = np.asarray(time, float)
    mask = np.isfinite(mag) & np.isfinite(time)
    if int(mask.sum()) < 10 or not np.isfinite(period) or float(period) <= 0:
        return out

    mag = mag[mask]
    time = time[mask]
    primary_idx = int(np.nanargmax(mag))
    epoch = float(time[primary_idx])
    phase = np.mod((time - epoch) / float(period), 1.0)
    baseline = float(np.nanpercentile(mag, 25.0))
    if not np.isfinite(baseline):
        return out

    def window_depth(center: float, half_width: float = 0.08) -> float:
        distance = np.abs(((phase - center + 0.5) % 1.0) - 0.5)
        local = mag[distance <= half_width]
        if local.size < 2:
            return np.nan
        eclipse_mag = float(np.nanpercentile(local, 90.0))
        return float(max(0.0, eclipse_mag - baseline)) if np.isfinite(eclipse_mag) else np.nan

    primary_depth = window_depth(0.0)
    secondary_depth = window_depth(0.5)
    out["eb_primary_min_depth"] = primary_depth
    out["eb_secondary_min_depth"] = secondary_depth
    depths = [d for d in (primary_depth, secondary_depth) if np.isfinite(d) and d > 0]
    if len(depths) == 2:
        out["eb_rminima"] = float(min(depths) / max(depths))
    return out


def _wrap_angle_2pi(angle: float) -> float:
    if not np.isfinite(angle):
        return np.nan
    return float(np.mod(angle, 2.0 * np.pi))


def _fourier_nan_result(max_harmonics: int) -> dict[str, float]:
    result: dict[str, float] = {
        "harmonics_order": np.nan,
        "harmonics_period": np.nan,
        "harmonics_a0": np.nan,
        "harmonics_model_amplitude": np.nan,
        "harmonics_reduced_chi2": np.nan,
        "harmonics_mse": np.nan,
    }
    for k in range(1, max_harmonics + 1):
        result[f"harmonics_mag_{k}"] = np.nan
        result[f"harmonics_a{k}"] = np.nan
        result[f"harmonics_b{k}"] = np.nan
    for k in range(2, max_harmonics + 1):
        result[f"harmonics_phase_{k}"] = np.nan
        result[f"harmonics_r{k}1"] = np.nan
    return result


def _build_fourier_design_matrix(phase: np.ndarray, n_harmonics: int) -> np.ndarray:
    design = np.ones((len(phase), 1 + 2 * n_harmonics), dtype=float)
    for k in range(1, n_harmonics + 1):
        angle = 2.0 * np.pi * k * phase
        design[:, 2 * k - 1] = np.cos(angle)
        design[:, 2 * k] = np.sin(angle)
    return design


def _solve_fourier_least_squares(
    mag: np.ndarray,
    design: np.ndarray,
    err: np.ndarray | None = None,
) -> dict[str, float | np.ndarray]:
    mag = np.asarray(mag, float)
    if err is not None:
        err = np.asarray(err, float)
        weights = np.where(np.isfinite(err) & (err > 0), 1.0 / np.square(err), np.nan)
        use_weights = np.isfinite(weights).all()
    else:
        weights = None
        use_weights = False

    try:
        if use_weights:
            sqrt_w = np.sqrt(weights)
            coeffs, _, _, _ = np.linalg.lstsq(design * sqrt_w[:, None], mag * sqrt_w, rcond=None)
        else:
            coeffs, _, _, _ = np.linalg.lstsq(design, mag, rcond=None)
    except np.linalg.LinAlgError:
        return {
            "coeffs": np.full(design.shape[1], np.nan, dtype=float),
            "fitted": np.full(len(mag), np.nan, dtype=float),
            "bic": np.nan,
            "mse": np.nan,
            "reduced_chi2": np.nan,
        }

    fitted = design @ coeffs
    resid = mag - fitted
    n = len(mag)
    n_params = design.shape[1]
    mse = float(np.mean(np.square(resid))) if n > 0 else np.nan

    if use_weights:
        chi2 = float(np.sum(np.square(resid / err)))
        bic = chi2 + n_params * np.log(max(n, 1))
        dof = n - n_params
        reduced_chi2 = chi2 / dof if dof > 0 else np.nan
    else:
        rss = float(np.sum(np.square(resid)))
        bic = n * np.log(max(rss / max(n, 1), 1e-12)) + n_params * np.log(max(n, 1))
        reduced_chi2 = np.nan

    return {
        "coeffs": coeffs,
        "fitted": fitted,
        "bic": float(bic),
        "mse": mse,
        "reduced_chi2": float(reduced_chi2) if np.isfinite(reduced_chi2) else np.nan,
    }


def _fit_phase_fourier_series(
    mag,
    time,
    period,
    err=None,
    max_harmonics: int = 7,
) -> dict[str, object] | None:
    mag = np.asarray(mag, float)
    time = np.asarray(time, float)
    if err is not None:
        err = np.asarray(err, float)
        mask = np.isfinite(mag) & np.isfinite(time) & np.isfinite(err) & (err > 0)
    else:
        mask = np.isfinite(mag) & np.isfinite(time)

    if (not np.isfinite(period)) or period <= 0:
        return None
    if int(mask.sum()) < 5:
        return None

    mag = mag[mask]
    time = time[mask]
    err = err[mask] if err is not None else None

    n = len(mag)
    max_order = min(int(max_harmonics), max((n - 2) // 2, 0))
    if max_order < 1:
        return None

    t0 = float(np.min(time))
    phase = np.mod((time - t0) / float(period), 1.0)

    fits: list[dict[str, float | np.ndarray]] = []
    for order in range(1, max_order + 1):
        design = _build_fourier_design_matrix(phase, order)
        fit = _solve_fourier_least_squares(mag, design, err=err)
        fit["order"] = order
        fits.append(fit)

    fits = [fit for fit in fits if np.isfinite(fit["bic"])]
    if not fits:
        return None

    return {
        "mag": mag,
        "time": time,
        "err": err,
        "phase": phase,
        "max_order": max_order,
        "fits": fits,
        "best_fit": min(fits, key=lambda fit: float(fit["bic"])),
        "full_fit": fits[-1],
    }


def fit_fourier_decomposition(mag, time, period, err=None, max_harmonics=7):
    """Fit a classical harmonic series to a phase-folded light curve.

    The returned amplitudes are the Fourier ``A_k`` terms. The returned
    ``harmonics_phase_k`` values are the classical phase combinations
    ``phi_k1 = phi_k - k * phi_1`` in a cosine-series convention, wrapped onto
    ``[0, 2pi)``.
    """
    result = _fourier_nan_result(max_harmonics)
    phase_fit = _fit_phase_fourier_series(mag, time, period, err=err, max_harmonics=max_harmonics)
    if phase_fit is None:
        return result

    best_fit = phase_fit["best_fit"]
    # Every returned coefficient and fit-quality value must describe the same
    # BIC-selected model.  Previously ``harmonics_order`` described the best
    # model while coefficients and chi-square came from the maximum order.
    max_order = int(best_fit["order"])
    coeffs = np.asarray(best_fit["coeffs"], float)

    result["harmonics_order"] = int(best_fit["order"])
    result["harmonics_period"] = float(period)
    result["harmonics_a0"] = float(coeffs[0])
    result["harmonics_reduced_chi2"] = float(best_fit["reduced_chi2"]) if np.isfinite(best_fit["reduced_chi2"]) else np.nan
    result["harmonics_mse"] = float(best_fit["mse"])

    amplitudes: dict[int, float] = {}
    phases_abs: dict[int, float] = {}
    for k in range(1, max_order + 1):
        cos_coeff = float(coeffs[2 * k - 1])
        sin_coeff = float(coeffs[2 * k])
        amplitudes[k] = float(np.hypot(cos_coeff, sin_coeff))
        phases_abs[k] = _wrap_angle_2pi(np.arctan2(-sin_coeff, cos_coeff))
        result[f"harmonics_mag_{k}"] = amplitudes[k]
        # Paper convention: a_k multiplies sin(2*pi*k*phase), b_k multiplies cos(...).
        result[f"harmonics_a{k}"] = sin_coeff
        result[f"harmonics_b{k}"] = cos_coeff

    amp1 = amplitudes.get(1, np.nan)
    phi1 = phases_abs.get(1, np.nan)
    for k in range(2, max_order + 1):
        phi_k1 = _wrap_angle_2pi(phases_abs[k] - k * phi1)
        result[f"harmonics_phase_{k}"] = phi_k1
        if np.isfinite(amp1) and amp1 > 0:
            result[f"harmonics_r{k}1"] = float(amplitudes[k] / amp1)

    phase_grid = np.linspace(0.0, 1.0, 1024, endpoint=False)
    model_grid = _build_fourier_design_matrix(phase_grid, max_order) @ coeffs
    result["harmonics_model_amplitude"] = float(np.nanmax(model_grid) - np.nanmin(model_grid))
    return result


def _empty_quasi_periodicity_result(
    status: str,
    *,
    n_points: int = 0,
    n_phase_bins: int = Q_TEMPLATE_N_PHASE_BINS,
    smooth_window_bins: int = Q_TEMPLATE_SMOOTH_WINDOW_BINS,
    raw_scatter: float = np.nan,
) -> dict[str, object]:
    return {
        "q": np.nan,
        "method": Q_TEMPLATE_METHOD,
        "n_points": int(n_points),
        "n_bins": int(n_phase_bins),
        "populated_bins": np.nan,
        "bin_coverage": np.nan,
        "smooth_window_bins": int(smooth_window_bins),
        "template_amplitude": np.nan,
        "raw_scatter": float(raw_scatter) if np.isfinite(raw_scatter) else np.nan,
        "resid_scatter": np.nan,
        "scatter_ratio": np.nan,
        "evaluation": Q_TEMPLATE_EVALUATION,
        "n_folds": 0,
        "status": str(status),
    }


def _circular_fill_template(template: np.ndarray) -> np.ndarray | None:
    values = np.asarray(template, dtype=float)
    n_bins = int(values.size)
    finite = np.isfinite(values)
    if n_bins == 0 or not finite.any():
        return None
    if finite.sum() == 1:
        return np.full(n_bins, float(values[finite][0]), dtype=float)

    centers = (np.arange(n_bins, dtype=float) + 0.5) / float(n_bins)
    xp = np.concatenate([centers[finite] - 1.0, centers[finite], centers[finite] + 1.0])
    fp = np.concatenate([values[finite], values[finite], values[finite]])
    return np.interp(centers, xp, fp)


def _circular_boxcar_smooth(values: np.ndarray, window_bins: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    window = max(int(window_bins), 1)
    if window <= 1:
        return arr.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.concatenate([arr[-pad:], arr, arr[:pad]])
    return np.convolve(padded, kernel, mode="valid")


def phase_template_quasi_periodicity(
    mag,
    time,
    err,
    period,
    *,
    n_phase_bins: int = Q_TEMPLATE_N_PHASE_BINS,
    min_bin_points: int = Q_TEMPLATE_MIN_BIN_POINTS,
    smooth_window_bins: int = Q_TEMPLATE_SMOOTH_WINDOW_BINS,
    min_bin_coverage: float = Q_TEMPLATE_MIN_BIN_COVERAGE,
    noise_subtract: bool = Q_TEMPLATE_NOISE_SUBTRACT,
) -> dict[str, object]:
    """Cody-style Q using an empirical phase-folded template."""
    try:
        period_value = float(period)
    except (TypeError, ValueError):
        period_value = np.nan
    if not np.isfinite(period_value) or period_value <= 0:
        return _empty_quasi_periodicity_result("invalid_period", n_phase_bins=n_phase_bins, smooth_window_bins=smooth_window_bins)

    mag = np.asarray(mag, float)
    time = np.asarray(time, float)
    err = np.asarray(err, float)
    mask = np.isfinite(mag) & np.isfinite(time) & np.isfinite(err) & (err > 0)
    n_points = int(mask.sum())
    if n_points < 5:
        return _empty_quasi_periodicity_result(
            "insufficient_points",
            n_points=n_points,
            n_phase_bins=n_phase_bins,
            smooth_window_bins=smooth_window_bins,
        )

    mag = mag[mask]
    time = time[mask]
    err = err[mask]
    raw_var = float(np.var(mag, ddof=1))
    raw_scatter = float(np.sqrt(raw_var)) if np.isfinite(raw_var) and raw_var >= 0 else np.nan
    noise_var = float(np.mean(np.square(err)))
    denom = raw_var - noise_var if bool(noise_subtract) else raw_var
    if not np.isfinite(denom) or denom <= 0:
        return _empty_quasi_periodicity_result(
            "low_intrinsic_variance",
            n_points=n_points,
            n_phase_bins=n_phase_bins,
            smooth_window_bins=smooth_window_bins,
            raw_scatter=raw_scatter,
        )

    n_bins = max(int(n_phase_bins), 4)
    phase = np.mod((time - float(np.min(time))) / period_value, 1.0)
    bin_idx = np.floor(phase * n_bins).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    template = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    for idx in range(n_bins):
        vals = mag[bin_idx == idx]
        if vals.size >= int(min_bin_points):
            template[idx] = float(np.median(vals))
            counts[idx] = int(vals.size)

    populated_bins = int(np.count_nonzero(np.isfinite(template)))
    bin_coverage = float(populated_bins / n_bins)
    min_populated_bins = max(8, int(np.ceil(float(min_bin_coverage) * n_bins)))
    if populated_bins < min_populated_bins:
        result = _empty_quasi_periodicity_result(
            "insufficient_phase_coverage",
            n_points=n_points,
            n_phase_bins=n_bins,
            smooth_window_bins=smooth_window_bins,
            raw_scatter=raw_scatter,
        )
        result["populated_bins"] = populated_bins
        result["bin_coverage"] = bin_coverage
        return result

    filled = _circular_fill_template(template)
    if filled is None or not np.isfinite(filled).any():
        result = _empty_quasi_periodicity_result(
            "template_fill_failed",
            n_points=n_points,
            n_phase_bins=n_bins,
            smooth_window_bins=smooth_window_bins,
            raw_scatter=raw_scatter,
        )
        result["populated_bins"] = populated_bins
        result["bin_coverage"] = bin_coverage
        return result

    smoothed = _circular_boxcar_smooth(filled, int(smooth_window_bins))
    centers = (np.arange(n_bins, dtype=float) + 0.5) / float(n_bins)
    xp = np.concatenate([centers - 1.0, centers, centers + 1.0])
    # Evaluate each point with a template that did not include its observing
    # cycle. This prevents Q from looking artificially good because the same
    # noisy point helped create the model used to score it.
    cycle = np.floor((time - float(np.min(time))) / period_value).astype(np.int64)
    unique_cycles = np.unique(cycle)
    n_folds = min(5, int(unique_cycles.size))
    model = np.full(mag.size, np.nan, dtype=float)
    if n_folds >= 2:
        fold_id = np.mod(cycle, n_folds)
        for fold in range(n_folds):
            train = fold_id != fold
            test = ~train
            if np.count_nonzero(train) < max(10, int(min_bin_points) * 4) or not np.any(test):
                continue
            train_template = np.full(n_bins, np.nan, dtype=float)
            for idx in range(n_bins):
                vals = mag[train & (bin_idx == idx)]
                if vals.size >= int(min_bin_points):
                    train_template[idx] = float(np.median(vals))
            train_filled = _circular_fill_template(train_template)
            if train_filled is None:
                continue
            train_smoothed = _circular_boxcar_smooth(train_filled, int(smooth_window_bins))
            train_fp = np.concatenate([train_smoothed, train_smoothed, train_smoothed])
            model[test] = np.interp(phase[test], xp, train_fp)
    else:
        # With fewer than two observed cycles there is no honest held-out
        # estimate of repeatability.
        result = _empty_quasi_periodicity_result(
            "insufficient_cycles",
            n_points=n_points,
            n_phase_bins=n_bins,
            smooth_window_bins=smooth_window_bins,
            raw_scatter=raw_scatter,
        )
        result["populated_bins"] = populated_bins
        result["bin_coverage"] = bin_coverage
        return result
    valid_model = np.isfinite(model) & np.isfinite(mag)
    if int(valid_model.sum()) < 2:
        result = _empty_quasi_periodicity_result(
            "template_model_failed",
            n_points=n_points,
            n_phase_bins=n_bins,
            smooth_window_bins=smooth_window_bins,
            raw_scatter=raw_scatter,
        )
        result["populated_bins"] = populated_bins
        result["bin_coverage"] = bin_coverage
        return result

    resid = mag[valid_model] - model[valid_model]
    resid_var = float(np.var(resid, ddof=1))
    resid_noise_var = float(np.mean(np.square(err[valid_model])))
    numer = resid_var - resid_noise_var if bool(noise_subtract) else resid_var
    q_value = float(numer / denom) if np.isfinite(numer) else np.nan
    resid_scatter = float(np.sqrt(resid_var)) if np.isfinite(resid_var) and resid_var >= 0 else np.nan
    scatter_ratio = float(resid_scatter / raw_scatter) if np.isfinite(resid_scatter) and np.isfinite(raw_scatter) and raw_scatter > 0 else np.nan
    template_amp = float(np.nanmax(smoothed) - np.nanmin(smoothed)) if np.isfinite(smoothed).any() else np.nan

    return {
        "q": q_value,
        "method": Q_TEMPLATE_METHOD,
        "n_points": int(valid_model.sum()),
        "n_bins": int(n_bins),
        "populated_bins": int(populated_bins),
        "bin_coverage": float(bin_coverage),
        "smooth_window_bins": int(smooth_window_bins),
        "template_amplitude": template_amp,
        "raw_scatter": raw_scatter,
        "resid_scatter": resid_scatter,
        "scatter_ratio": scatter_ratio,
        "evaluation": Q_TEMPLATE_EVALUATION,
        "n_folds": int(n_folds),
        "status": "ok" if np.isfinite(q_value) else "invalid_q",
    }


def quasi_periodicity_metric(mag, time, err, period, max_harmonics=7) -> float:
    """Cody-style quasi-periodicity Q from an empirical phase-folded template.

    ``max_harmonics`` is accepted for compatibility with older callers; Q no
    longer uses a Fourier model.
    """
    result = phase_template_quasi_periodicity(mag, time, err, period)
    try:
        return float(result["q"])
    except (TypeError, ValueError, KeyError):
        return np.nan


def psi_cs(mag, time, period):
    """Range of cumulative sum on the phase-folded light curve."""
    mag = np.asarray(mag, float)
    time = np.asarray(time, float)
    mask = np.isfinite(mag) & np.isfinite(time)
    if mask.sum() < 3 or not np.isfinite(period) or period <= 0:
        return np.nan
    mag = mag[mask]
    time = time[mask]

    phase = (time / period) % 1.0
    order = np.argsort(phase)
    mag_sorted = mag[order]
    return rcs(mag_sorted)


def psi_eta(mag, time, period):
    """Eta_e on the phase-folded light curve."""
    mag = np.asarray(mag, float)
    time = np.asarray(time, float)
    mask = np.isfinite(mag) & np.isfinite(time)
    if mask.sum() < 3 or not np.isfinite(period) or period <= 0:
        return np.nan
    mag = mag[mask]
    time = time[mask]

    phase = (time / period) % 1.0
    order = np.argsort(phase)
    mag_sorted = mag[order]
    return von_neumann_ratio(mag_sorted)


def fit_drw(jd, mag, mag_err):
    """Fit an Ornstein-Uhlenbeck (true DRW) GP and return (RMS, tau days).

    The covariance is ``sigma**2 * exp(-|dt| / tau)``.  This uses a
    celerite ``RealTerm``; the formerly used fixed-Q SHO kernel is not a DRW.
    Returns ``(NaN, NaN)`` if the fit is not measurable or has <20 points.
    """
    jd = np.asarray(jd, float)
    mag = np.asarray(mag, float)
    mag_err = np.asarray(mag_err, float)
    mask = np.isfinite(jd) & np.isfinite(mag) & np.isfinite(mag_err) & (mag_err > 0)
    if mask.sum() < 20:
        return np.nan, np.nan

    order = np.argsort(jd[mask], kind="stable")
    t = jd[mask][order]
    mag_fit = mag[mask][order]
    mag_err_fit = mag_err[mask][order]

    # Subtract mean for numerical stability
    mean_mag = np.mean(mag_fit)
    mag_fit = mag_fit - mean_mag

    # Initial guesses
    var = np.var(mag_fit)
    tau0 = (t[-1] - t[0]) / 10.0
    if tau0 <= 0 or var <= 0:
        return np.nan, np.nan

    def neg_log_like(params):
        log_variance, log_inv_tau = params
        variance = np.exp(log_variance)
        inv_tau = np.exp(log_inv_tau)
        kernel = _cterms.RealTerm(a=variance, c=inv_tau)
        gp = _GP(kernel)
        gp.compute(t, diag=mag_err_fit**2)
        return -gp.log_likelihood(mag_fit)

    try:
        positive_dt = np.diff(t)
        positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0)]
        min_tau = max(float(np.median(positive_dt)) if positive_dt.size else 1e-3, 1e-3)
        max_tau = max(float(t[-1] - t[0]) * 10.0, min_tau * 10.0)
        x0 = np.array([np.log(var), np.log(1.0 / tau0)])
        bounds = [
            (np.log(max(var * 1e-6, 1e-12)), np.log(max(var * 1e6, 1e-6))),
            (np.log(1.0 / max_tau), np.log(1.0 / min_tau)),
        ]
        result = minimize(neg_log_like, x0, method="L-BFGS-B", bounds=bounds)
        if not result.success:
            return np.nan, np.nan
        log_variance, log_inv_tau = result.x
        tau = 1.0 / np.exp(log_inv_tau)
        sigma = np.sqrt(np.exp(log_variance))
        if not (np.isfinite(sigma) and np.isfinite(tau) and sigma > 0 and tau > 0):
            return np.nan, np.nan
        return float(sigma), float(tau)
    except Exception:
        return np.nan, np.nan


# ---------------------------------------------------------------------------
# IAR_phi: Irregular Autoregressive coefficient
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# IAR_phi: Irregular Autoregressive coefficient
# ---------------------------------------------------------------------------
_HAS_IAR = True


def iar_phi_fit(jd, mag, err):
    """Fit an IAR(1) model and return phi.

    phi ~ 1 means smooth slow variability, phi ~ 0 means uncorrelated noise.
    Returns NaN if fit fails or < 10 points.
    """
    if not _HAS_IAR:
        return np.nan

    jd = np.asarray(jd, float)
    mag = np.asarray(mag, float)
    err = np.asarray(err, float)
    mask = np.isfinite(jd) & np.isfinite(mag) & np.isfinite(err) & (err > 0)
    if mask.sum() < 10:
        return np.nan

    t = jd[mask]
    y = mag[mask] - np.mean(mag[mask])
    yerr = err[mask]

    try:

        result = minimize_scalar(
            lambda phi: float(_IARphikalman(phi, y, yerr, t, zero_mean=False)),
            bounds=(1e-6, 1 - 1e-6),
            method="bounded",
        )
        phi = float(result.x)
        return phi if np.isfinite(phi) else np.nan
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# MHPS: Mexican Hat Power Spectrum (wavelet variance at two timescales)
# ---------------------------------------------------------------------------
def mhps(jd, mag, err):
    """Mexican Hat Power Spectrum features at 10d and 100d timescales.

    Returns dict with keys: mhps_high, mhps_low, mhps_non_zero,
    mhps_pn_flag, mhps_ratio.  All NaN if < 20 points.
    """
    nan_result = {
        "mhps_high": np.nan,
        "mhps_low": np.nan,
        "mhps_non_zero": np.nan,
        "mhps_pn_flag": np.nan,
        "mhps_ratio": np.nan,
    }

    jd = np.asarray(jd, float)
    mag = np.asarray(mag, float)
    err = np.asarray(err, float)
    mask = np.isfinite(jd) & np.isfinite(mag) & np.isfinite(err)
    if mask.sum() < 20:
        return nan_result

    t = jd[mask]
    y = mag[mask]
    e = err[mask]
    y = y - np.mean(y)

    def wavelet_variance(scale):
        """Compute Mexican Hat wavelet variance at given scale (days)."""
        n = len(t)
        coeffs = np.zeros(n)
        for i in range(n):
            dt = (t - t[i]) / scale
            # Mexican Hat (Ricker) wavelet: (1 - t^2) * exp(-t^2/2)
            psi = (1.0 - dt**2) * np.exp(-0.5 * dt**2)
            # Normalize
            norm = np.sum(psi**2)
            if norm > 0:
                coeffs[i] = np.sum(y * psi) / np.sqrt(norm)
        return float(np.var(coeffs))

    scale_short = 10.0   # days
    scale_long = 100.0   # days

    try:
        var_high = wavelet_variance(scale_short)
        var_low = wavelet_variance(scale_long)
        n_non_zero = int(mask.sum())
        mean_err_sq = float(np.mean(e**2))
        pn_flag = 1.0 if mean_err_sq > var_high else 0.0
        ratio = var_low / var_high if var_high > 0 else np.nan

        return {
            "mhps_high": var_high,
            "mhps_low": var_low,
            "mhps_non_zero": float(n_non_zero),
            "mhps_pn_flag": pn_flag,
            "mhps_ratio": ratio,
        }
    except Exception:
        return nan_result


def compute_stats(
    asassn_id,
    path,
    use_only_good=True,
    drop_dupes=True,
    use_g: bool | None = None,
    compute_ls=False,
    file_ext: str | None = None,
    feature_period_days: float | None = None,
    feature_period_source: str | None = None,
    input_frame: pd.DataFrame | None = None,
):
    """Compute light-curve statistics from aligned ASAS-SN photometry.

    By default (``use_g=None``), g and V observations are treated as one time
    series after shifting the V-band median onto the g-band median.  Explicit
    ``use_g=True`` and ``use_g=False`` retain the legacy g-only and V-only
    modes, respectively, including fallback to the other band when the
    requested band is empty.
    """

    if input_frame is not None:
        loaded = input_frame.copy()
        if "v_g_band" in loaded.columns:
            band = pd.to_numeric(loaded["v_g_band"], errors="coerce")
            df_g_raw = loaded.loc[band != 1].copy()
            df_v_raw = loaded.loc[band == 1].copy()
        else:
            df_g_raw, df_v_raw = loaded, pd.DataFrame()
    else:
        df_g_raw, df_v_raw = _load_stats_lightcurve_frames(
            asassn_id,
            path,
            file_ext=file_ext,
        )

    df_g = _prepare_stats_lightcurve_frame(df_g_raw)
    df_v = _prepare_stats_lightcurve_frame(df_v_raw)
    sokolovsky_g_input = df_g.copy()
    sokolovsky_v_input = df_v.copy()
    if not sokolovsky_g_input.empty:
        sokolovsky_g_input["v_g_band"] = 0
    if not sokolovsky_v_input.empty:
        sokolovsky_v_input["v_g_band"] = 1
    sokolovsky_input = pd.concat(
        [frame for frame in (sokolovsky_g_input, sokolovsky_v_input) if not frame.empty],
        ignore_index=True,
    ) if (not sokolovsky_g_input.empty or not sokolovsky_v_input.empty) else pd.DataFrame(columns=_LC_COLUMNS)
    _sokolovsky_frame, sokolovsky_summary = compute_sokolovsky_peak_to_peak_summary(
        asassn_id,
        path,
        use_only_good=use_only_good,
        drop_dupes=drop_dupes,
        input_frame=sokolovsky_input,
    )

    if use_g is None:
        frames = [frame for frame in (df_g, df_v) if not frame.empty]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=_LC_COLUMNS)
    elif use_g:
        if df_g.empty and not df_v.empty:
            print(f"[warn] {asassn_id}: g-band empty; using V-band instead.")
            df = df_v.copy()
        else:
            df = df_g.copy()
    else:
        if df_v.empty and not df_g.empty:
            print(f"[warn] {asassn_id}: V-band empty; using g-band instead.")
            df = df_g.copy()
        else:
            df = df_v.copy()

    base_n = len(df)
    df = _filter_stats_lightcurve_frame(
        df,
        use_only_good=use_only_good,
        drop_dupes=drop_dupes,
        duplicate_subset=("JD", "v_g_band", "camera#", "camera_name", "field")
        if use_g is None else ("JD", "camera#", "camera_name", "field"),
    )

    band_values = pd.to_numeric(df.get("v_g_band"), errors="coerce")
    g_points = int((band_values == 0).sum())
    v_points = int((band_values == 1).sum())
    v_minus_g_offset = np.nan
    band_alignment = "none"
    if use_g is None and g_points > 0 and v_points > 0:
        df_aligned, v_minus_g_offset, band_alignment = _align_v_to_g_with_overlap_policy(df)
        if np.isfinite(v_minus_g_offset):
            df = df_aligned
        else:
            # Combined scalar statistics are unsafe without a measured color
            # offset. Use the better sampled band while retaining both bands
            # separately for camera/band-normalized Q below.
            preferred_band = 0 if g_points >= v_points else 1
            df = df.loc[band_values == preferred_band].copy()

    if g_points > 0 and v_points > 0:
        band_mode = "g+V" if np.isfinite(v_minus_g_offset) else "g+V_unaligned_single_band_stats"
    elif g_points > 0:
        band_mode = "g"
    elif v_points > 0:
        band_mode = "V"
    else:
        band_mode = "none"

    df = df.sort_values("JD").reset_index(drop=True)
    kept_n = len(df)
    if df.empty:
        return df, OrderedDict([
            ("compute_status", "insufficient_data"),
            ("compute_error", "no usable photometry after quality filtering"),
            ("photometry_band_mode", band_mode),
            ("photometry_band_alignment", band_alignment),
            ("photometry_g_points", g_points),
            ("photometry_v_points", v_points),
            ("photometry_v_minus_g_offset_mag", np.nan),
            ("file_points_total", int(base_n)),
            ("file_points_kept_after_filter", 0),
        ])
    field_summary = compute_field_summary(df)

    q_frames = [frame for frame in (df_g, df_v) if not frame.empty]
    q_df = pd.concat(q_frames, ignore_index=True) if q_frames else pd.DataFrame(columns=_LC_COLUMNS)
    q_df = _filter_stats_lightcurve_frame(
        q_df,
        use_only_good=use_only_good,
        drop_dupes=drop_dupes,
        duplicate_subset=("JD", "v_g_band", "camera#", "camera_name", "field"),
    )

    # time axis in days since first exposure (JD is in days already)
    jd0 = df["JD"].iloc[0]
    df["t_days"] = df["JD"] - jd0

    # cadence (diffs)
    df["dt_days"] = df["t_days"].diff()
    dt = df["dt_days"].iloc[1:].values

    # 3-day exposures: binned and rolling
    df["bin3d"] = np.floor(df["t_days"]/3).astype("Int64")
    per3d_binned = df.groupby("bin3d").size().rename("count").reset_index()

    # sliding window (previous 3 days)
    t = df["t_days"].values
    counts_rolling = np.empty(len(t), dtype=int)
    j = 0
    for i in range(len(t)):
        while j < i and t[j] < t[i] - 3.0:
            j += 1
        counts_rolling[i] = i - j + 1
    df["count_in_prev_3d"] = counts_rolling

    # exposures per night & duty cycle
    # treat "night" as floor(JD)
    df["night"] = np.floor(df["JD"]).astype(int)
    per_night = df.groupby("night").size().rename("n_exp").reset_index()
    nights_observed = len(per_night)
    span_days = df["t_days"].iloc[-1] - df["t_days"].iloc[0]
    # duty cycle ~ fraction of nights with >=1 obs over total span in nights
    total_nights_in_span = int(np.floor(df["JD"].iloc[-1]) - np.floor(df["JD"].iloc[0]) + 1)
    duty_cycle = nights_observed / total_nights_in_span if total_nights_in_span > 0 else np.nan

    # "seasons": long gaps (> 30 days) split segments
    gap_threshold = 30.0
    cut_idx = np.where(dt > gap_threshold)[0]  # indices of large gaps (relative to df[1:])
    segments = []
    start = 0
    for c in cut_idx:
        end = c + 1
        segments.append((start, end))
        start = end
    segments.append((start, len(df)))
    seasons = []
    for (a,b) in segments:
        sub = df.iloc[a:b]
        seasons.append({
            "start_JD": float(sub["JD"].iloc[0]),
            "end_JD": float(sub["JD"].iloc[-1]),
            "n_obs": int(len(sub)),
            "span_days": float(sub["t_days"].iloc[-1] - sub["t_days"].iloc[0]),
        })

    # largest gaps (top 10)
    gaps = df.loc[df["dt_days"].nlargest(10).index, ["JD","t_days","dt_days"]].copy()
    gaps["JD_prev"] = df["JD"].shift(1).loc[gaps.index].values
    gaps = gaps.sort_values("dt_days", ascending=False).reset_index(drop=True)

    # photometric stats
    mag = df["mag"].values
    merr = df["error"].values
    w = 1.0 / np.where(merr>0, merr**2, np.nan)
    mean_mag = float(np.nanmean(mag))
    median_mag = float(np.nanmedian(mag))
    wmean_mag, wsem_mag = weighted_mean(mag, w)
    wstd_mag = float(weighted_std(mag, merr))
    std_mag = float(np.nanstd(mag, ddof=1))
    rsig_mag = float(robust_sigma(mag))
    iqr_mag = float(paper_iqr(mag))
    p05, p16, p84, p95 = [pct(mag, q) for q in [5,16,84,95]]

    # clipped stats)
    med = np.nanmedian(mag)
    rs = robust_sigma(mag)
    if np.isfinite(rs) and rs > 0:
        clip_mask = np.abs(mag - med) <= 3*rs
    else:
        clip_mask = np.isfinite(mag)
    mag_clip = mag[clip_mask]
    mean_clip = float(np.nanmean(mag_clip))
    std_clip  = float(np.nanstd(mag_clip, ddof=1))
    n_outliers = int(np.size(mag) - np.size(mag_clip))

    snr = SNR_CONVERSION_FACTOR / merr
    err_stats = {
        "error_mean": float(np.nanmean(merr)),
        "error_median": float(np.nanmedian(merr)),
        "error_p05": pct(merr,5),
        "error_p95": pct(merr,95),
        "snr_median": float(np.nanmedian(snr)),
        "snr_p05": pct(snr,5),
        "snr_p95": pct(snr,95),
    }

    # variability diagnostics vs constant model
    model = wmean_mag if np.isfinite(wmean_mag) else median_mag
    rchisq = float(reduced_chisq(mag, merr, model))
    vnr    = float(inverse_von_neumann_ratio(mag))
    ac1    = float(lag1_autocorr(mag))
    slope_d_per_day, intercept, r2 = linear_trend(df["t_days"].values, mag)
    slope_d_per_year = slope_d_per_day * 365.25 if np.isfinite(slope_d_per_day) else np.nan
    roms = float(roms_statistic(mag, merr))
    sokolovsky_v = float(sokolovsky_summary["variability_sokolovsky_v"])
    stetson = paper_stetson_indices(df["JD"].values, mag, merr)
    flux_asymmetry_m = flux_asymmetry_metric(mag)
    string_length_stats = baseline_subtracted_string_length(df)
    ls_stats = lomb_scargle_summary(df["JD"].values, mag, merr) if compute_ls else {
        "ls_best_period_days": np.nan,
        "ls_peak_power": np.nan,
        "ls_fap": np.nan,
    }

    # ALeRCE-style features
    jd_arr = df["JD"].values
    _amplitude = amplitude(mag)
    _beyond1std = beyond_1_std(mag)
    _con = con(mag)
    _delta_mag = delta_mag(mag)
    _intrinsic_sigma_mag = intrinsic_sigma_mag(mag, merr)
    _first_mag = float(mag[0]) if mag.size > 0 else np.nan
    _gskew = gskew(mag)
    _max_slope = max_slope(mag, jd_arr)
    _meanvariance = meanvariance(mag)
    _median_abs_dev = median_abs_dev(mag)
    _median_brp = median_brp(mag)
    _percent_amplitude = percent_amplitude(mag)
    _ahl_ratio = ahl_ratio(mag)
    _q31 = q31(mag)
    _skew = skew(mag)
    _small_kurtosis = small_kurtosis(mag)
    _constancy_p_value = constancy_p_value(mag, merr)
    _anderson_darling = anderson_darling(mag)
    _pair_slope_trend = pair_slope_trend(mag)
    _rcs = rcs(mag)
    _autocor_length = autocor_length(mag, vnr)
    _sf_amplitude, _sf_gamma = structure_function(mag, jd_arr)

    # Period-dependent features use an explicit period. Lomb-Scargle remains
    # an optional legacy feature, but no longer drives Q or related stats by
    # default.
    try:
        best_period = float(feature_period_days)
    except (TypeError, ValueError):
        best_period = np.nan
    if not np.isfinite(best_period) or best_period <= 0:
        best_period = np.nan
        periodic_feature_source = ""
    else:
        periodic_feature_source = str(feature_period_source or "explicit_period")
    q_time, q_mag, q_err = _camera_band_normalized_q_arrays(q_df)
    q_result, best_period, periodic_feature_source = _phase_template_quasi_periodicity_best_period(
        q_mag,
        q_time,
        q_err,
        best_period,
        feature_period_source=periodic_feature_source,
    )
    quasi_periodicity_q = q_result["q"]
    _harmonics = fit_fourier_decomposition(mag, jd_arr, best_period, err=merr)
    _psi_cs = psi_cs(mag, jd_arr, best_period)
    _psi_eta = psi_eta(mag, jd_arr, best_period)
    _lk_stats = lafler_kinman_period_stats(jd_arr, mag, best_period)
    _window_alias = window_function_alias_peaks(jd_arr)
    _eb_minima = eb_minima_ratio(mag, jd_arr, best_period)

    # stochastic model features
    _drw_sigma, _drw_tau = fit_drw(jd_arr, mag, merr)
    _iar_phi = iar_phi_fit(jd_arr, mag, merr)
    _mhps = mhps(jd_arr, mag, merr)

    # per camera/field/band usage + offsets and scatter
    global_med = median_mag
    def per_group_stats(group, name):
        out = group.agg(
            n_obs=("mag","size"),
            med_mag=("mag","median"),
            mad_sigma=("mag", lambda x: robust_sigma(x)),
            mean_err=("error","mean"),
        ).reset_index()
        out["offset_vs_global_med"] = out["med_mag"] - global_med
        out = out.sort_values("n_obs", ascending=False).reset_index(drop=True)
        out.attrs["group_name"] = name
        return out


    by_camera  = per_group_stats(df.groupby("camera_name"), "camera")

    # Add LOO metrics to by_camera
    try:
        loo_res = compute_camera_loo_metrics(df, cam_col="camera_name")
        if loo_res:
            by_camera["loo_corr"] = by_camera["camera_name"].map(lambda c: loo_res.get(c, {}).get("corr", np.nan))
            by_camera["loo_offset"] = by_camera["camera_name"].map(lambda c: loo_res.get(c, {}).get("offset", np.nan))
            by_camera["loo_rms"] = by_camera["camera_name"].map(lambda c: loo_res.get(c, {}).get("rms", np.nan))
            
            # Aggregate to top-level stats
            loo_corr_valid = by_camera["loo_corr"].dropna()
            loo_rms_valid = by_camera["loo_rms"].dropna()
            camera_loo_corr_min = loo_corr_valid.min() if not loo_corr_valid.empty else np.nan
            camera_loo_corr_median = loo_corr_valid.median() if not loo_corr_valid.empty else np.nan
            camera_loo_rms_max = loo_rms_valid.max() if not loo_rms_valid.empty else np.nan
        else:
            by_camera["loo_corr"] = np.nan
            by_camera["loo_offset"] = np.nan
            by_camera["loo_rms"] = np.nan
            camera_loo_corr_min = np.nan
            camera_loo_corr_median = np.nan
            camera_loo_rms_max = np.nan
    except Exception as e:
        by_camera["loo_corr"] = np.nan
        by_camera["loo_offset"] = np.nan
        by_camera["loo_rms"] = np.nan
        camera_loo_corr_min = np.nan
        camera_loo_corr_median = np.nan
        camera_loo_rms_max = np.nan
        print(f"[warn] LOO metrics failed: {e}")

    by_field   = per_group_stats(df.groupby("field"), "field")
    by_band    = per_group_stats(df.groupby("v_g_band"), "v_g_band")
    by_camera_and_field = per_group_stats(df.groupby(["camera_name","field"]), "camera_and_field")

    # cadence distributions per camera
    def per_cam_cadence(d):
        d = d.sort_values("t_days")
        dt = d["t_days"].diff().iloc[1:].values
        return pd.Series({
            "dt_median": np.nanmedian(dt) if dt.size else np.nan,
            "dt_mean":   np.nanmean(dt) if dt.size else np.nan,
            "dt_p05":    pct(dt,5) if dt.size else np.nan,
            "dt_p95":    pct(dt,95) if dt.size else np.nan,
        })
    cadence_by_camera = df.groupby("camera_name").apply(per_cam_cadence, include_groups=False).reset_index()

    # nightly stats table (exposures and median mag per night)
    nightly = df.groupby("night").agg(
        n_exp=("mag","size"),
        med_mag=("mag","median"),
        med_err=("error","median"),
    ).reset_index()

    # package summary
    summary = OrderedDict([
        ("compute_status", "ok"),
        ("compute_error", ""),
        ("photometry_band_mode", band_mode),
        ("photometry_band_alignment", band_alignment),
        ("photometry_g_points", g_points),
        ("photometry_v_points", v_points),
        ("photometry_v_minus_g_offset_mag", float(v_minus_g_offset)),
        ("file_points_total", int(base_n)),
        ("file_points_kept_after_filter", int(kept_n)),
        ("jd_start", float(df["JD"].iloc[0])),
        ("jd_end", float(df["JD"].iloc[-1])),
        ("time_span_days", float(span_days)),
        ("n_unique_nights", int(nights_observed)),
        ("duty_cycle_fraction", float(duty_cycle)),
        ("cadence_mean_dt_days", float(np.nanmean(dt)) if dt.size else np.nan),
        ("cadence_median_dt_days", float(np.nanmedian(dt)) if dt.size else np.nan),
        ("cadence_p05_dt_days", pct(dt,5) if dt.size else np.nan),
        ("cadence_p95_dt_days", pct(dt,95) if dt.size else np.nan),
        ("largest_gaps_top10_days", gaps[["JD_prev","JD","dt_days"]].rename(columns={"dt_days":"gap_days"})),
        ("exposures_per_3d_binned", per3d_binned),
        ("rolling_count_prev_3d_at_each_obs", df[["JD","t_days","count_in_prev_3d"]]),
        ("photometry_mean_mag", mean_mag),
        ("photometry_median_mag", median_mag),
        ("photometry_weighted_mean_mag", float(wmean_mag)),
        ("photometry_weighted_mean_sem", float(wsem_mag)),
        ("photometry_weighted_std_mag", wstd_mag),
        ("photometry_std_mag", std_mag),
        ("photometry_robust_sigma_mag", rsig_mag),
        ("photometry_IQR_mag", iqr_mag),
        ("photometry_p05_mag", p05),
        ("photometry_p16_mag", p16),
        ("photometry_p84_mag", p84),
        ("photometry_p95_mag", p95),
        ("clipped_mean_mag_3sigma_about_median", mean_clip),
        ("clipped_std_mag_3sigma_about_median", std_clip),
        ("n_outliers_removed_robust_3sigma", n_outliers),
        ("error_and_snr_stats", err_stats),
        ("variability_reduced_chi2_vs_constant", rchisq),
        ("variability_von_neumann_ratio", vnr),
        ("variability_roms", roms),
        ("variability_sokolovsky_v", sokolovsky_v),
        ("variability_lag1_autocorr", ac1),
        ("variability_stetson_I", stetson["stetson_I"]),
        ("variability_stetson_J", stetson["stetson_J"]),
        ("variability_stetson_K", stetson["stetson_K"]),
        ("variability_stetson_L", stetson["stetson_L"]),
        ("variability_stetson_J_time", stetson["stetson_J_time"]),
        ("variability_stetson_L_time", stetson["stetson_L_time"]),
        ("variability_flux_asymmetry_m", flux_asymmetry_m),
        ("variability_quasi_periodicity_q", quasi_periodicity_q),
        ("variability_quasi_periodicity_method", q_result["method"]),
        ("variability_quasi_periodicity_n_points", q_result["n_points"]),
        ("variability_quasi_periodicity_n_bins", q_result["n_bins"]),
        ("variability_quasi_periodicity_populated_bins", q_result["populated_bins"]),
        ("variability_quasi_periodicity_bin_coverage", q_result["bin_coverage"]),
        ("variability_quasi_periodicity_smooth_window_bins", q_result["smooth_window_bins"]),
        ("variability_quasi_periodicity_template_amplitude", q_result["template_amplitude"]),
        ("variability_quasi_periodicity_raw_scatter", q_result["raw_scatter"]),
        ("variability_quasi_periodicity_resid_scatter", q_result["resid_scatter"]),
        ("variability_quasi_periodicity_scatter_ratio", q_result["scatter_ratio"]),
        ("variability_quasi_periodicity_evaluation", q_result["evaluation"]),
        ("variability_quasi_periodicity_n_folds", q_result["n_folds"]),
        ("variability_quasi_periodicity_status", q_result["status"]),
        ("variability_periodic_feature_period_days", best_period),
        ("variability_periodic_feature_period_source", periodic_feature_source),
        ("variability_string_length_resid_total", string_length_stats["string_length_total"]),
        ("variability_string_length_resid_mean_step", string_length_stats["string_length_mean_step"]),
        ("variability_string_length_resid_n_steps", string_length_stats["string_length_n_steps"]),
        ("variability_lomb_scargle_best_period_days", ls_stats["ls_best_period_days"]),
        ("variability_lomb_scargle_peak_power", ls_stats["ls_peak_power"]),
        ("variability_lomb_scargle_fap", ls_stats["ls_fap"]),
        ("trend_slope_mag_per_day", slope_d_per_day),
        ("trend_slope_mag_per_year", slope_d_per_year),
        ("trend_r2", r2),
        # ALeRCE-style features
        ("amplitude", _amplitude),
        ("beyond_1_std", _beyond1std),
        ("con", _con),
        ("delta_mag_fid", _delta_mag),
        ("intrinsic_sigma_mag", _intrinsic_sigma_mag),
        ("first_mag", _first_mag),
        ("gskew", _gskew),
        ("max_slope", _max_slope),
        ("meanvariance", _meanvariance),
        ("median_abs_dev", _median_abs_dev),
        ("median_brp", _median_brp),
        ("percent_amplitude", _percent_amplitude),
        ("ahl_ratio", _ahl_ratio),
        ("q31", _q31),
        ("skew", _skew),
        ("small_kurtosis", _small_kurtosis),
        ("constancy_p_value", _constancy_p_value),
        ("anderson_darling", _anderson_darling),
        ("pair_slope_trend", _pair_slope_trend),
        ("rcs", _rcs),
        ("autocor_length", _autocor_length),
        ("sf_ml_amplitude", _sf_amplitude),
        ("sf_ml_gamma", _sf_gamma),
        # period-dependent features
        ("harmonics_order", _harmonics["harmonics_order"]),
        ("harmonics_period", _harmonics["harmonics_period"]),
        ("harmonics_a0", _harmonics["harmonics_a0"]),
        ("harmonics_model_amplitude", _harmonics["harmonics_model_amplitude"]),
        ("harmonics_reduced_chi2", _harmonics["harmonics_reduced_chi2"]),
        ("harmonics_mag_1", _harmonics["harmonics_mag_1"]),
        ("harmonics_mag_2", _harmonics["harmonics_mag_2"]),
        ("harmonics_mag_3", _harmonics["harmonics_mag_3"]),
        ("harmonics_mag_4", _harmonics["harmonics_mag_4"]),
        ("harmonics_mag_5", _harmonics["harmonics_mag_5"]),
        ("harmonics_mag_6", _harmonics["harmonics_mag_6"]),
        ("harmonics_mag_7", _harmonics["harmonics_mag_7"]),
        ("harmonics_a1", _harmonics["harmonics_a1"]),
        ("harmonics_a2", _harmonics["harmonics_a2"]),
        ("harmonics_a3", _harmonics["harmonics_a3"]),
        ("harmonics_a4", _harmonics["harmonics_a4"]),
        ("harmonics_a5", _harmonics["harmonics_a5"]),
        ("harmonics_a6", _harmonics["harmonics_a6"]),
        ("harmonics_a7", _harmonics["harmonics_a7"]),
        ("harmonics_b1", _harmonics["harmonics_b1"]),
        ("harmonics_b2", _harmonics["harmonics_b2"]),
        ("harmonics_b3", _harmonics["harmonics_b3"]),
        ("harmonics_b4", _harmonics["harmonics_b4"]),
        ("harmonics_b5", _harmonics["harmonics_b5"]),
        ("harmonics_b6", _harmonics["harmonics_b6"]),
        ("harmonics_b7", _harmonics["harmonics_b7"]),
        ("harmonics_r21", _harmonics["harmonics_r21"]),
        ("harmonics_r31", _harmonics["harmonics_r31"]),
        ("harmonics_r41", _harmonics["harmonics_r41"]),
        ("harmonics_r51", _harmonics["harmonics_r51"]),
        ("harmonics_r61", _harmonics["harmonics_r61"]),
        ("harmonics_r71", _harmonics["harmonics_r71"]),
        ("harmonics_phase_2", _harmonics["harmonics_phase_2"]),
        ("harmonics_phase_3", _harmonics["harmonics_phase_3"]),
        ("harmonics_phase_4", _harmonics["harmonics_phase_4"]),
        ("harmonics_phase_5", _harmonics["harmonics_phase_5"]),
        ("harmonics_phase_6", _harmonics["harmonics_phase_6"]),
        ("harmonics_phase_7", _harmonics["harmonics_phase_7"]),
        ("harmonics_mse", _harmonics["harmonics_mse"]),
        ("psi_cs", _psi_cs),
        ("psi_eta", _psi_eta),
        ("lafler_kinman_t_time", _lk_stats["lafler_kinman_t_time"]),
        ("lafler_kinman_t_phase", _lk_stats["lafler_kinman_t_phase"]),
        ("lafler_kinman_delta", _lk_stats["lafler_kinman_delta"]),
        ("window_alias_period_1", _window_alias["window_alias_period_1"]),
        ("window_alias_power_1", _window_alias["window_alias_power_1"]),
        ("window_alias_period_2", _window_alias["window_alias_period_2"]),
        ("window_alias_power_2", _window_alias["window_alias_power_2"]),
        ("window_alias_period_3", _window_alias["window_alias_period_3"]),
        ("window_alias_power_3", _window_alias["window_alias_power_3"]),
        ("window_alias_period_4", _window_alias["window_alias_period_4"]),
        ("window_alias_power_4", _window_alias["window_alias_power_4"]),
        ("window_alias_period_5", _window_alias["window_alias_period_5"]),
        ("window_alias_power_5", _window_alias["window_alias_power_5"]),
        ("eb_rminima", _eb_minima["eb_rminima"]),
        ("eb_primary_min_depth", _eb_minima["eb_primary_min_depth"]),
        ("eb_secondary_min_depth", _eb_minima["eb_secondary_min_depth"]),
        # stochastic model features
        ("gp_drw_sigma", _drw_sigma),
        ("gp_drw_tau", _drw_tau),
        ("gp_drw_model", "ornstein_uhlenbeck_realterm_v1"),
        ("iar_phi", _iar_phi),
        ("mhps_high", _mhps["mhps_high"]),
        ("mhps_low", _mhps["mhps_low"]),
        ("mhps_non_zero", _mhps["mhps_non_zero"]),
        ("mhps_pn_flag", _mhps["mhps_pn_flag"]),
        ("mhps_ratio", _mhps["mhps_ratio"]),
        ("camera_loo_corr_min", camera_loo_corr_min),
        ("camera_loo_corr_median", camera_loo_corr_median),
        ("camera_loo_rms_max", camera_loo_rms_max),
        ("asassn_field_key", field_summary["asassn_field_key"]),
        ("asassn_fields", field_summary["asassn_fields"]),
        ("asassn_field_count", field_summary["asassn_field_count"]),
        ("asassn_field_key_fraction", field_summary["asassn_field_key_fraction"]),
        ("camera_name_key", field_summary["camera_name_key"]),
        ("camera_names", field_summary["camera_names"]),
        ("camera_name_count", field_summary["camera_name_count"]),
        ("camera_name_key_fraction", field_summary["camera_name_key_fraction"]),
        ("by_camera", by_camera),
        ("by_field", by_field),
        ("by_band", by_band),
        ("by_camera_and_field", by_camera_and_field),
        ("cadence_by_camera", cadence_by_camera),
        ("nightly_table", nightly),
        ("seasons", pd.DataFrame(seasons)),
    ])
    summary.update(compute_derived_feature_row(summary))

    return df, summary

def print_summary(summary, max_rows=10):
    def headframe(x, n=max_rows):
        return x.head(n).to_string(index=False)

    def _fmt(v, d=4):
        return f"{v:.{d}f}" if np.isfinite(v) else "NaN"

    print("\n=== CORE TIMING ===")
    print(f"JD start/end: {summary['jd_start']:.6f} → {summary['jd_end']:.6f}  | span: {summary['time_span_days']:.2f} d")
    print(f"Unique nights: {summary['n_unique_nights']}  | Duty cycle: {summary['duty_cycle_fraction']:.3f}")

    print("\n=== CADENCE (Δt in days) ===")
    print(f"mean={summary['cadence_mean_dt_days']:.3f}  median={summary['cadence_median_dt_days']:.3f}  p05={summary['cadence_p05_dt_days']:.3f}  p95={summary['cadence_p95_dt_days']:.3f}")
    print("Top 10 gaps (days):")
    print(headframe(summary["largest_gaps_top10_days"][["JD_prev","JD","gap_days"]]))

    print("\n=== EXPOSURES PER 3 DAYS ===")
    print("Non-overlapping bins (first rows):")
    print(headframe(summary["exposures_per_3d_binned"]))
    print("Rolling count at each obs (first rows):")
    print(headframe(summary["rolling_count_prev_3d_at_each_obs"]))

    print("\n=== PHOTOMETRY ===")
    print(f"mean={summary['photometry_mean_mag']:.6f}  median={summary['photometry_median_mag']:.6f}  wmean={summary['photometry_weighted_mean_mag']:.6f}±{summary['photometry_weighted_mean_sem']:.6f}")
    print(f"std={summary['photometry_std_mag']:.6f}  wstd={summary['photometry_weighted_std_mag']:.6f}  robust_sigma={summary['photometry_robust_sigma_mag']:.6f}  IQR={summary['photometry_IQR_mag']:.6f}")
    print(f"p05={summary['photometry_p05_mag']:.6f}  p16={summary['photometry_p16_mag']:.6f}  p84={summary['photometry_p84_mag']:.6f}  p95={summary['photometry_p95_mag']:.6f}")
    print(f"clipped mean={summary['clipped_mean_mag_3sigma_about_median']:.6f}  clipped std={summary['clipped_std_mag_3sigma_about_median']:.6f}  outliers={summary['n_outliers_removed_robust_3sigma']}")

    es = summary["error_and_snr_stats"]
    print("\n=== ERRORS / SNR ===")
    print(f"error: mean={es['error_mean']:.6f}  median={es['error_median']:.6f}  p05={es['error_p05']:.6f}  p95={es['error_p95']:.6f}")
    print(f"SNR: median={es['snr_median']:.2f}  p05={es['snr_p05']:.2f}  p95={es['snr_p95']:.2f}")

    print("\n=== VARIABILITY / TREND ===")
    print(f"reduced χ² vs constant={summary['variability_reduced_chi2_vs_constant']:.3f}  | 1/η={summary['variability_von_neumann_ratio']:.3f}  | RoMS={summary['variability_roms']:.3f}  | lag-1 ρ={summary['variability_lag1_autocorr']:.3f}")
    print(
        f"Stetson I/J/K/L={summary['variability_stetson_I']:.3f} / {summary['variability_stetson_J']:.3f} / "
        f"{summary['variability_stetson_K']:.3f} / {summary['variability_stetson_L']:.3f}"
    )
    print(f"Stetson J(time)/L(time)={summary['variability_stetson_J_time']:.3f} / {summary['variability_stetson_L_time']:.3f}")
    sl_steps = summary.get("variability_string_length_resid_n_steps", np.nan)
    sl_steps_text = str(int(sl_steps)) if np.isfinite(sl_steps) else "NaN"
    print(
        "String length |Δresid| "
        f"total={_fmt(summary.get('variability_string_length_resid_total', np.nan))} "
        f"mean_step={_fmt(summary.get('variability_string_length_resid_mean_step', np.nan))} "
        f"n_steps={sl_steps_text}"
    )
    if "variability_lomb_scargle_best_period_days" in summary:
        print(f"Lomb-Scargle: best_period_days={summary['variability_lomb_scargle_best_period_days']:.6f}  peak_power={summary['variability_lomb_scargle_peak_power']:.6f}  fap={summary['variability_lomb_scargle_fap']:.3e}")
    print(f"trend slope={summary['trend_slope_mag_per_day']:.6e} mag/day ({summary['trend_slope_mag_per_year']:.6e} mag/yr),  R²={summary['trend_r2']:.3f}")

    print("\n=== ALeRCE FEATURES ===")
    print(f"Amplitude={_fmt(summary['amplitude'])}  Beyond1Std={_fmt(summary['beyond_1_std'])}  Con={summary['con']}  delta_mag={_fmt(summary['delta_mag_fid'])}")
    print(f"IntrinsicSigmaMag={_fmt(summary['intrinsic_sigma_mag'])}  first_mag={_fmt(summary['first_mag'],3)}  Gskew={_fmt(summary['gskew'])}  MaxSlope={_fmt(summary['max_slope'])}")
    print(f"Meanvariance={_fmt(summary['meanvariance'])}  MedianAbsDev={_fmt(summary['median_abs_dev'])}  MedianBRP={_fmt(summary['median_brp'])}  PercentAmplitude={_fmt(summary['percent_amplitude'])}")
    print(f"Q31={_fmt(summary['q31'])}  Skew={_fmt(summary['skew'])}  SmallKurtosis={_fmt(summary['small_kurtosis'])}  ConstancyPValue={_fmt(summary['constancy_p_value'])}")
    print(f"AndersonDarling={_fmt(summary['anderson_darling'])}  PairSlopeTrend={_fmt(summary['pair_slope_trend'])}  Rcs={_fmt(summary['rcs'])}  Autocor_length={summary['autocor_length']}")
    print(f"SF_ML_amplitude={_fmt(summary['sf_ml_amplitude'])}  SF_ML_gamma={_fmt(summary['sf_ml_gamma'])}")

    print("\n=== HARMONICS (folded LC) ===")
    print(
        f"order={_fmt(summary['harmonics_order'], 0)}  "
        f"period={_fmt(summary['harmonics_period'], 6)} d  "
        f"A0={_fmt(summary['harmonics_a0'])}  "
        f"model_amp={_fmt(summary['harmonics_model_amplitude'])}  "
        f"red_chi2={_fmt(summary['harmonics_reduced_chi2'])}"
    )
    h_mags = "  ".join(f"H{k}={_fmt(summary[f'harmonics_mag_{k}'])}" for k in range(1, 8))
    print(f"Mag: {h_mags}")
    h_ratios = "  ".join(f"R{k}1={_fmt(summary[f'harmonics_r{k}1'])}" for k in range(2, 8))
    print(f"Ratios: {h_ratios}")
    h_phases = "  ".join(f"φ{k}1={_fmt(summary[f'harmonics_phase_{k}'])}" for k in range(2, 8))
    print(f"Phase: {h_phases}")
    print(f"MSE={_fmt(summary['harmonics_mse'])}  Psi_CS={_fmt(summary['psi_cs'])}  Psi_eta={_fmt(summary['psi_eta'])}")

    print("\n=== STOCHASTIC MODELS ===")
    print(f"GP_DRW: sigma={_fmt(summary['gp_drw_sigma'])}  tau={_fmt(summary['gp_drw_tau'])} d")
    print(f"IAR: phi={_fmt(summary['iar_phi'])}")
    print(f"MHPS: high={_fmt(summary['mhps_high'])}  low={_fmt(summary['mhps_low'])}  ratio={_fmt(summary['mhps_ratio'])}  PN_flag={summary['mhps_pn_flag']}  non_zero={summary['mhps_non_zero']}")

    print("\n=== BY CAMERA (top) ===")
    print(headframe(summary["by_camera"]))
    print("\n=== BY FIELD (top) ===")
    print(headframe(summary["by_field"]))
    print("\n=== BY BAND (all) ===")
    print(headframe(summary["by_band"]))
    print("\n=== BY CAMERA+FIELD (top) ===")
    print(headframe(summary["by_camera_and_field"]))

    print("\n=== CADENCE BY CAMERA ===")
    print(headframe(summary["cadence_by_camera"]))

    print("\n=== NIGHTLY TABLE (first rows) ===")
    print(headframe(summary["nightly_table"]))

    print("\n=== SEASONS (gap > 30 d defines a new season) ===")
    print(summary["seasons"].to_string(index=False))

def load_dat(path, has_header=False):
    # auto-handle comments and variable whitespace
    names = ["JD",
            "mag",
            "error",
            "good_bad",
            "camera#",
            "v_g_band",
            "saturated",
            "cam_field"]
    kw = dict(sep=r"\s+", comment="#", engine="python")
    if has_header:
        df = pd.read_csv(path, **kw)
        if len(df.columns) < len(names):
            # pad names if fewer columns
            df.columns = names[:len(df.columns)]
    else:
        df = pd.read_csv(path, header=None, names=names, **kw)

    if "cam_field" in df.columns:
        split = df["cam_field"].astype("string").str.split("/", n=1, expand=True)
        if split.shape[1] >= 1:
            df["camera_name"] = split[0].fillna("").astype(str).str.strip()
        if split.shape[1] >= 2:
            df["field"] = split[1].fillna("").astype(str).str.strip()
        df = df.drop(columns=["cam_field"])

    return df


def _period_feature_from_row(row: dict[str, object]) -> tuple[float | None, str | None]:
    for key in (
        "periodicity_period",
        "pdm_corrected_period",
        "ce_corrected_period",
        "pdm_period",
        "ce_period",
        "period_consensus_days",
        "pre_periodicity_selected_period",
        "phase_period_days",
    ):
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            return float(value), key
    return None, None


def _q_arbitrated_period_source(feature_period_source: str | None, factor: object) -> str:
    source = str(feature_period_source or "explicit_period")
    try:
        factor_value = float(factor)
    except (TypeError, ValueError):
        return source
    if not np.isfinite(factor_value) or np.isclose(factor_value, 1.0):
        return source
    return f"{source}:q_factor_{factor_value:g}"


def _phase_template_quasi_periodicity_best_period(
    mag: np.ndarray,
    time: np.ndarray,
    err: np.ndarray,
    period: float | None,
    *,
    feature_period_source: str | None = None,
) -> tuple[dict[str, object], float, str]:
    try:
        base_period = float(period)
    except (TypeError, ValueError):
        base_period = np.nan
    if not np.isfinite(base_period) or base_period <= 0:
        return (
            phase_template_quasi_periodicity(mag, time, err, np.nan),
            np.nan,
            "",
        )

    factors_list: list[float] = []
    for factor_raw in Q_PERIOD_ARBITRATION_FACTORS:
        try:
            factor = float(factor_raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(factor) and factor > 0:
            factors_list.append(factor)
    factors = tuple(factors_list)
    if not factors:
        return (
            phase_template_quasi_periodicity(mag, time, err, base_period),
            float(base_period),
            str(feature_period_source or "explicit_period"),
        )

    candidates = native_harmonic_period_candidates(
        base_period,
        min_period=float(base_period) * min(factors),
        max_period=float(base_period) * max(factors),
        harmonic_factors=Q_PERIOD_ARBITRATION_FACTORS,
    )
    base_result: dict[str, object] | None = None
    base_candidate: dict[str, object] | None = None

    for candidate in candidates:
        candidate_period = float(candidate["period"])
        result = phase_template_quasi_periodicity(mag, time, err, candidate_period)
        candidate["_q_result"] = result
        candidate["q_status"] = result.get("status")
        try:
            q_value = float(result.get("q"))
        except (TypeError, ValueError):
            q_value = np.nan
        if np.isfinite(q_value):
            candidate["selection_objective"] = float(q_value)
        if np.isclose(float(candidate.get("factor", np.nan)), 1.0):
            base_result = result
            base_candidate = candidate

    selected = choose_native_harmonic_candidate(
        candidates,
        objective_key="selection_objective",
        min_rel_improvement=Q_PERIOD_ARBITRATION_MIN_REL_IMPROVEMENT,
        upward_min_rel_improvement=Q_PERIOD_ARBITRATION_UPWARD_MIN_REL_IMPROVEMENT,
    )
    if selected is not None and isinstance(selected.get("_q_result"), dict):
        selected_factor = selected.get("factor", 1.0)
        return (
            selected["_q_result"],
            float(selected["period"]),
            _q_arbitrated_period_source(feature_period_source, selected_factor),
        )

    if base_result is None:
        base_result = phase_template_quasi_periodicity(mag, time, err, base_period)
        base_candidate = {"factor": 1.0}
    return (
        base_result,
        float(base_period),
        _q_arbitrated_period_source(feature_period_source, base_candidate.get("factor", 1.0)),
    )


def compute_quasi_periodicity_summary(
    asassn_id,
    path,
    *,
    use_only_good: bool = True,
    drop_dupes: bool = True,
    file_ext: str | None = None,
    feature_period_days: float | None = None,
    feature_period_source: str | None = None,
) -> OrderedDict:
    """Compute only the phase-template Q fields from a light curve.

    This is the lightweight refresh path for review products that already have
    current non-Q stats and only need the native-period Q semantics updated.
    """
    df_g_raw, df_v_raw = _load_stats_lightcurve_frames(
        asassn_id,
        path,
        file_ext=file_ext,
    )

    df_g = _prepare_stats_lightcurve_frame(df_g_raw)
    df_v = _prepare_stats_lightcurve_frame(df_v_raw)
    q_frames = [frame for frame in (df_g, df_v) if not frame.empty]
    q_df = pd.concat(q_frames, ignore_index=True) if q_frames else pd.DataFrame(columns=_LC_COLUMNS)
    q_df = _filter_stats_lightcurve_frame(
        q_df,
        use_only_good=use_only_good,
        drop_dupes=drop_dupes,
        duplicate_subset=("JD", "v_g_band", "camera#", "camera_name", "field"),
    )

    try:
        best_period = float(feature_period_days)
    except (TypeError, ValueError):
        best_period = np.nan
    if not np.isfinite(best_period) or best_period <= 0:
        best_period = np.nan
        periodic_feature_source = ""
    else:
        periodic_feature_source = str(feature_period_source or "explicit_period")

    q_time, q_mag, q_err = _camera_band_normalized_q_arrays(q_df)
    q_result, best_period, periodic_feature_source = _phase_template_quasi_periodicity_best_period(
        q_mag,
        q_time,
        q_err,
        best_period,
        feature_period_source=periodic_feature_source,
    )
    return OrderedDict(
        [
            ("variability_quasi_periodicity_q", q_result["q"]),
            ("variability_quasi_periodicity_method", q_result["method"]),
            ("variability_quasi_periodicity_n_points", q_result["n_points"]),
            ("variability_quasi_periodicity_n_bins", q_result["n_bins"]),
            ("variability_quasi_periodicity_populated_bins", q_result["populated_bins"]),
            ("variability_quasi_periodicity_bin_coverage", q_result["bin_coverage"]),
            ("variability_quasi_periodicity_smooth_window_bins", q_result["smooth_window_bins"]),
            ("variability_quasi_periodicity_template_amplitude", q_result["template_amplitude"]),
            ("variability_quasi_periodicity_raw_scatter", q_result["raw_scatter"]),
            ("variability_quasi_periodicity_resid_scatter", q_result["resid_scatter"]),
            ("variability_quasi_periodicity_scatter_ratio", q_result["scatter_ratio"]),
            ("variability_quasi_periodicity_evaluation", q_result["evaluation"]),
            ("variability_quasi_periodicity_n_folds", q_result["n_folds"]),
            ("variability_quasi_periodicity_status", q_result["status"]),
            ("variability_periodic_feature_period_days", best_period),
            ("variability_periodic_feature_period_source", periodic_feature_source),
        ]
    )


def _enrich_row_worker(args: tuple) -> dict:
    """Top-level picklable worker for parallel compute_stats enrichment.

    Args:
        args: (row_dict, asassn_id, path, compute_ls[, file_ext]).  When the
            row contains ``lc_path``, that exact path is authoritative and the
            positional ID/path values are retained only for legacy callers.

    Returns:
        Row dict with flattened stats_* columns merged in, or the original
        row dict unchanged if compute_stats raises.
    """
    if len(args) >= 5:
        row_dict, asassn_id, stats_path, compute_ls, file_ext = args[:5]
    else:
        row_dict, asassn_id, stats_path, compute_ls = args
        file_ext = None
    try:
        raw_lc_path = row_dict.get("lc_path")
        if raw_lc_path is not None and raw_lc_path is not pd.NA:
            lc_path_text = str(raw_lc_path).strip()
            if lc_path_text.lower() not in {"", "nan", "none", "null", "<na>"}:
                exact_lc_path = Path(lc_path_text).expanduser()
                asassn_id = exact_lc_path.stem
                stats_path = str(exact_lc_path)
                file_ext = exact_lc_path.suffix.lstrip(".") or file_ext
        feature_period_days, feature_period_source = _period_feature_from_row(row_dict)
        _, stats_dict = compute_stats(
            asassn_id,
            stats_path,
            use_only_good=True,
            compute_ls=compute_ls,
            file_ext=file_ext,
            feature_period_days=feature_period_days,
            feature_period_source=feature_period_source,
        )
        merged = dict(row_dict)
        for k, v in stats_dict.items():
            if k in {"compute_status", "compute_error"}:
                merged[f"stats_{k}"] = v
                continue
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    col = f"stats_{k}_{sub_k}"
                    if col not in merged:
                        merged[col] = sub_v
            elif isinstance(v, (pd.DataFrame, pd.Series)):
                continue
            elif str(k).startswith("derived_"):
                if k not in merged:
                    merged[k] = v
            elif f"stats_{k}" not in merged:
                merged[f"stats_{k}"] = v
        for k, v in compute_derived_feature_row(merged).items():
            if k not in merged:
                merged[k] = v
        layered = to_layer_first_frame(pd.DataFrame([merged]), run_derived=False)
        if not layered.empty:
            merged.update(layered.iloc[0].to_dict())
        return merged
    except Exception as exc:
        failed = dict(row_dict)
        failed["stats_compute_status"] = "error"
        failed["stats_compute_error"] = f"{type(exc).__name__}: {exc}"
        return failed


def main():
    ap = argparse.ArgumentParser(description="Compute rich stats for a photometry .dat file.")
    ap.add_argument("path", help="path to .dat file")
    ap.add_argument("--include-all", action="store_true", help="do NOT filter by good_bad==1 & saturated==0")
    ap.add_argument("--keep-dupes",   action="store_true", help="keep duplicate JD rows instead of dropping")
    ap.add_argument("--has-header",   action="store_true", help="file has a header row")
    ap.add_argument("--lomb-scargle", action="store_true", help="compute Lomb-Scargle periodogram summary stats")
    args = ap.parse_args()

    input_path = Path(args.path).expanduser()
    df = load_dat(input_path, has_header=args.has_header)
    df2, summary = compute_stats(
        input_path.stem,
        input_path.parent,
        use_only_good=not args.include_all,
        drop_dupes=not args.keep_dupes,
        compute_ls=args.lomb_scargle,
        file_ext=input_path.suffix.lstrip(".") or None,
        input_frame=df,
    )
    if summary.get("compute_status") != "ok":
        print(
            f"Statistics unavailable: {summary.get('compute_status', 'error')} - "
            f"{summary.get('compute_error', '')}"
        )
        return
    print_summary(summary)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python stats.py yourfile.dat [--include-all] [--keep-dupes] [--has-header]")
    else:
        main()
