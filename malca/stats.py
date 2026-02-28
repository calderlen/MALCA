"""
Outputs:
- Core timing/cadence stats (including 3-day exposure metrics and largest gaps)
- Photometric stats (weighted/unweighted/clipped/MAD/IQR/percentiles)
- Quality & error stats (SNR dist, fractions by good/saturated)
- Variability diagnostics (reduced chisq, von Neumann ratio, lag-1 autocorr, trend slope, Stetson I/J/K)
- Optional Lomb-Scargle periodogram summary stats
- Nightly/seasonal coverage & duty cycle
- Per-camera / per-field / per-band usage + offsets and scatter
"""

import sys, io, argparse, math
from collections import OrderedDict
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from astropy.timeseries import LombScargle

from malca.utils import read_lc_dat2, read_lc_csv

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
    # 1.4826 * MAD (median absolute deviation)
    return 1.4826 * np.median(np.abs(x - np.median(x)))

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
    chi2 = np.sum((resid / err) ** 2)
    sigma2 = chi2 / n
    return float(n * np.log(sigma2) + n_params * np.log(n))

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

def lag1_autocorr(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3: return np.nan
    x0 = x[:-1] - x[:-1].mean()
    x1 = x[1:]  - x[1:].mean()
    den = np.sqrt(np.sum(x0**2) * np.sum(x1**2))
    return float(np.sum(x0 * x1) / den) if den > 0 else np.nan

def stetson_indices(mag, err):
    mag = np.asarray(mag, float)
    err = np.asarray(err, float)
    mask = np.isfinite(mag) & np.isfinite(err) & (err > 0)
    if mask.sum() < 2:
        return {"stetson_I": np.nan, "stetson_J": np.nan, "stetson_K": np.nan}

    m = mag[mask]
    e = err[mask]
    n = len(m)

    w = 1.0 / np.square(e)
    mu, _ = weighted_mean(m, w)
    if not np.isfinite(mu):
        mu = float(np.nanmedian(m))

    d = np.sqrt(n / (n - 1.0)) * (m - mu) / e
    if d.size < 2:
        return {"stetson_I": np.nan, "stetson_J": np.nan, "stetson_K": np.nan}

    P = d[:-1] * d[1:]
    stetson_I = float(np.sum(P))
    stetson_J = float(np.mean(np.sign(P) * np.sqrt(np.abs(P)))) if P.size else np.nan

    denom = np.sqrt(np.mean(d**2)) if d.size else np.nan
    if not np.isfinite(denom) or denom <= 0:
        stetson_K = np.nan
    else:
        stetson_K = float((1.0 / 0.798) * np.mean(np.abs(d)) / denom)

    return {"stetson_I": stetson_I, "stetson_J": stetson_J, "stetson_K": stetson_K}

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


def bootstrap_lomb_scargle(
    jd: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    min_frequency: float = 1.0 / 365.25,
    max_frequency: float = 10.0,
    exclude_alias_periods: bool = True,
    alias_tolerance: float = 0.1,
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
    if LombScargle is None:
        return {
            "ls_power": np.nan,
            "ls_period_days": np.nan,
            "ls_bootstrap_sig": np.nan,
            "ls_is_alias": False,
            "ls_is_significant": False,
        }

    jd = np.asarray(jd, float)
    mag = np.asarray(mag, float)
    err = np.asarray(err, float)

    mask = np.isfinite(jd) & np.isfinite(mag) & np.isfinite(err) & (err > 0)
    if mask.sum() < 50:
        return {
            "ls_power": np.nan,
            "ls_period_days": np.nan,
            "ls_bootstrap_sig": np.nan,
            "ls_is_alias": False,
            "ls_is_significant": False,
        }

    jd = jd[mask]
    mag = mag[mask]
    err = err[mask]

    # Known alias periods (sidereal day, half-day, lunar month, year, half-year)
    alias_periods = [1.0, 0.5, 29.53, 365.25, 182.625]

    try:
        ls = LombScargle(jd, mag, err)
        freq, power_spec = ls.autopower(minimum_frequency=min_frequency, maximum_frequency=max_frequency)

        if power_spec.size == 0:
            return {
                "ls_power": np.nan,
                "ls_period_days": np.nan,
                "ls_bootstrap_sig": np.nan,
                "ls_is_alias": False,
                "ls_is_significant": False,
            }

        max_idx = int(np.argmax(power_spec))
        ls_power = float(power_spec[max_idx])
        best_period = float(1.0 / freq[max_idx]) if freq[max_idx] > 0 else np.nan

        # Bootstrap significance
        bootstrap_powers = np.empty(n_bootstrap)
        rng = np.random.default_rng()
        for i in range(n_bootstrap):
            shuffled_mag = rng.permutation(mag)
            ls_boot = LombScargle(jd, shuffled_mag, err)
            _, power_boot = ls_boot.autopower(minimum_frequency=min_frequency, maximum_frequency=max_frequency)
            bootstrap_powers[i] = np.max(power_boot) if power_boot.size > 0 else 0.0

        bootstrap_sig = float(np.sum(bootstrap_powers >= ls_power) / n_bootstrap)

        # Check for alias periods
        is_alias = False
        if exclude_alias_periods and np.isfinite(best_period):
            is_alias = any(abs(best_period - ap) < alias_tolerance for ap in alias_periods)

        # Significant if bootstrap sig < 1% and not an alias
        is_significant = (bootstrap_sig < 0.01) and (not is_alias)

        return {
            "ls_power": ls_power,
            "ls_period_days": best_period,
            "ls_bootstrap_sig": bootstrap_sig,
            "ls_is_alias": is_alias,
            "ls_is_significant": is_significant,
        }

    except Exception:
        return {
            "ls_power": np.nan,
            "ls_period_days": np.nan,
            "ls_bootstrap_sig": np.nan,
            "ls_is_alias": False,
            "ls_is_significant": False,
        }

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


def excess_var(mag, err):
    """Intrinsic variability amplitude: sqrt((Var - mean(err^2)) / mean^2)."""
    mag = np.asarray(mag, float)
    err = np.asarray(err, float)
    mask = np.isfinite(mag) & np.isfinite(err) & (err > 0)
    if mask.sum() < 3:
        return np.nan
    mag = mag[mask]
    err = err[mask]
    mu = np.mean(mag)
    if mu == 0:
        return np.nan
    var = np.var(mag, ddof=1)
    mean_err2 = np.mean(err ** 2)
    inner = (var - mean_err2) / (mu ** 2)
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


def pvar(mag, err):
    """Probability that the source is variable (1 - chi2 CDF)."""
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
    if mag.size < 8:
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
    log_sf = np.log10(np.array(bin_sf))
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


def fit_harmonics(mag, time, period, n_harmonics=7):
    """
    Fit a Fourier/harmonic series to the phase-folded light curve.

    Returns dict with:
    - harmonics_mag_1..N: amplitudes of each harmonic
    - harmonics_phase_2..N: phases relative to fundamental
    - harmonics_mse: mean squared error of the fit
    """
    nan_result = {}
    for k in range(1, n_harmonics + 1):
        nan_result[f"harmonics_mag_{k}"] = np.nan
    for k in range(2, n_harmonics + 1):
        nan_result[f"harmonics_phase_{k}"] = np.nan
    nan_result["harmonics_mse"] = np.nan

    mag = np.asarray(mag, float)
    time = np.asarray(time, float)
    mask = np.isfinite(mag) & np.isfinite(time)
    if mask.sum() < 2 * n_harmonics + 1 or not np.isfinite(period) or period <= 0:
        return nan_result

    mag = mag[mask]
    time = time[mask]

    phase = (time / period) % 1.0

    # design matrix: [1, cos(2pi*phase), sin(2pi*phase), cos(4pi*phase), ...]
    n = len(mag)
    X = np.ones((n, 1 + 2 * n_harmonics))
    for k in range(1, n_harmonics + 1):
        X[:, 2 * k - 1] = np.cos(2 * np.pi * k * phase)
        X[:, 2 * k] = np.sin(2 * np.pi * k * phase)

    try:
        coeffs, residuals, _, _ = np.linalg.lstsq(X, mag, rcond=None)
    except np.linalg.LinAlgError:
        return nan_result

    result = {}
    phase_1 = np.arctan2(coeffs[2], coeffs[1])  # phase of fundamental

    for k in range(1, n_harmonics + 1):
        a_k = coeffs[2 * k - 1]
        b_k = coeffs[2 * k]
        result[f"harmonics_mag_{k}"] = float(np.sqrt(a_k ** 2 + b_k ** 2))

    for k in range(2, n_harmonics + 1):
        a_k = coeffs[2 * k - 1]
        b_k = coeffs[2 * k]
        phase_k = np.arctan2(b_k, a_k)
        # phase relative to fundamental
        result[f"harmonics_phase_{k}"] = float(phase_k - k * phase_1)

    fitted = X @ coeffs
    result["harmonics_mse"] = float(np.mean((mag - fitted) ** 2))

    return result


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


# ---------------------------------------------------------------------------
# GP_DRW: Damped Random Walk via celerite2
# ---------------------------------------------------------------------------
try:
    from celerite2 import GaussianProcess as _GP, terms as _cterms
    _HAS_CELERITE2 = True
except Exception:
    _GP = None
    _cterms = None
    _HAS_CELERITE2 = False


def fit_drw(jd, mag, err):
    """Fit a Damped Random Walk GP model and return (sigma, tau).

    Uses celerite2 SHOTerm with Q = 1/sqrt(2) (the DRW limit).
    Returns (NaN, NaN) if fit fails or < 20 points.
    """
    jd = np.asarray(jd, float)
    mag = np.asarray(mag, float)
    err = np.asarray(err, float)
    mask = np.isfinite(jd) & np.isfinite(mag) & np.isfinite(err) & (err > 0)
    if mask.sum() < 20 or not _HAS_CELERITE2:
        return np.nan, np.nan

    t = jd[mask]
    y = mag[mask]
    yerr = err[mask]

    # Subtract mean for numerical stability
    y_mean = np.mean(y)
    y = y - y_mean

    # Initial guesses
    var = np.var(y)
    tau0 = (t[-1] - t[0]) / 10.0
    if tau0 <= 0 or var <= 0:
        return np.nan, np.nan

    Q = 1.0 / np.sqrt(2.0)
    w0_init = 1.0 / tau0
    S0_init = var * tau0

    from scipy.optimize import minimize

    def neg_log_like(params):
        log_S0, log_w0 = params
        S0 = np.exp(log_S0)
        w0 = np.exp(log_w0)
        kernel = _cterms.SHOTerm(S0=S0, w0=w0, Q=Q)
        gp = _GP(kernel)
        gp.compute(t, diag=yerr**2)
        return -gp.log_likelihood(y)

    try:
        x0 = np.array([np.log(S0_init), np.log(w0_init)])
        result = minimize(neg_log_like, x0, method="L-BFGS-B")
        if not result.success:
            return np.nan, np.nan
        log_S0, log_w0 = result.x
        S0 = np.exp(log_S0)
        w0 = np.exp(log_w0)
        tau = 1.0 / w0
        sigma = np.sqrt(S0 / tau)
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
try:
    from iar.IARModel import IARphikalman as _IARphikalman
    _HAS_IAR = True
except Exception:
    _IARphikalman = None
    _HAS_IAR = False


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
        from scipy.optimize import minimize_scalar
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


def compute_stats(asassn_id, path, use_only_good=True, drop_dupes=True, use_g=True, compute_ls=False):

    df_g, df_v = read_lc_csv(asassn_id, path)
    if df_g.empty and df_v.empty:
        df_g, df_v = read_lc_dat2(asassn_id, path)

    if use_g:
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
    
    cols = ["JD",
            "mag",
            "error",
            "good_bad",
            "camera#",
            "v_g_band",
            "saturated",
            "camera_name",
            "field"]
    
    df.columns = cols[:len(df.columns)] + [f"extra_{i}" for i in range(len(df.columns)-len(cols))]

    for c in ["JD","mag","error","good_bad","camera#","v_g_band","saturated"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["JD","mag","error"]).sort_values("JD").reset_index(drop=True)

    # drop duplicate JDs
    if drop_dupes:
        df = df[~df["JD"].duplicated(keep="first")].reset_index(drop=True)

    # filtering
    base_n = len(df)
    if use_only_good:
        df = df[(df["good_bad"] == 1) & (df["saturated"] == 0)].reset_index(drop=True)
    kept_n = len(df)

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
    std_mag = float(np.nanstd(mag, ddof=1))
    rsig_mag = float(robust_sigma(mag))
    iqr_mag = float(np.nanpercentile(mag, 75) - np.nanpercentile(mag, 25))
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

    # error and SNR stats (SNR ~= 1.0857 / err )
    snr = 1.0857 / merr
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
    vnr    = float(von_neumann_ratio(mag))
    ac1    = float(lag1_autocorr(mag))
    slope_d_per_day, intercept, r2 = linear_trend(df["t_days"].values, mag)
    slope_d_per_year = slope_d_per_day * 365.25 if np.isfinite(slope_d_per_day) else np.nan
    stetson = stetson_indices(mag, merr)
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
    _excess_var = excess_var(mag, merr)
    _first_mag = float(mag[0]) if mag.size > 0 else np.nan
    _gskew = gskew(mag)
    _max_slope = max_slope(mag, jd_arr)
    _meanvariance = meanvariance(mag)
    _median_abs_dev = median_abs_dev(mag)
    _median_brp = median_brp(mag)
    _percent_amplitude = percent_amplitude(mag)
    _q31 = q31(mag)
    _skew = skew(mag)
    _small_kurtosis = small_kurtosis(mag)
    _pvar = pvar(mag, merr)
    _anderson_darling = anderson_darling(mag)
    _pair_slope_trend = pair_slope_trend(mag)
    _rcs = rcs(mag)
    _autocor_length = autocor_length(mag, vnr)
    _sf_amplitude, _sf_gamma = structure_function(mag, jd_arr)

    # period-dependent features (use LS best period)
    best_period = ls_stats["ls_best_period_days"]
    _harmonics = fit_harmonics(mag, jd_arr, best_period)
    _psi_cs = psi_cs(mag, jd_arr, best_period)
    _psi_eta = psi_eta(mag, jd_arr, best_period)

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
    by_field   = per_group_stats(df.groupby("field"), "field")
    by_band    = per_group_stats(df.groupby("v_g_band"), "v_g_band")
    by_camfld  = per_group_stats(df.groupby(["camera_name","field"]), "camera_field")

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
    cadence_by_camera = df.groupby("camera_name").apply(per_cam_cadence).reset_index()

    # nightly stats table (exposures and median mag per night)
    nightly = df.groupby("night").agg(
        n_exp=("mag","size"),
        med_mag=("mag","median"),
        med_err=("error","median"),
    ).reset_index()

    # package summary
    summary = OrderedDict([
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
        ("variability_lag1_autocorr", ac1),
        ("variability_stetson_I", stetson["stetson_I"]),
        ("variability_stetson_J", stetson["stetson_J"]),
        ("variability_stetson_K", stetson["stetson_K"]),
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
        ("excess_var", _excess_var),
        ("first_mag", _first_mag),
        ("gskew", _gskew),
        ("max_slope", _max_slope),
        ("meanvariance", _meanvariance),
        ("median_abs_dev", _median_abs_dev),
        ("median_brp", _median_brp),
        ("percent_amplitude", _percent_amplitude),
        ("q31", _q31),
        ("skew", _skew),
        ("small_kurtosis", _small_kurtosis),
        ("pvar", _pvar),
        ("anderson_darling", _anderson_darling),
        ("pair_slope_trend", _pair_slope_trend),
        ("rcs", _rcs),
        ("autocor_length", _autocor_length),
        ("sf_ml_amplitude", _sf_amplitude),
        ("sf_ml_gamma", _sf_gamma),
        # period-dependent features
        ("harmonics_mag_1", _harmonics["harmonics_mag_1"]),
        ("harmonics_mag_2", _harmonics["harmonics_mag_2"]),
        ("harmonics_mag_3", _harmonics["harmonics_mag_3"]),
        ("harmonics_mag_4", _harmonics["harmonics_mag_4"]),
        ("harmonics_mag_5", _harmonics["harmonics_mag_5"]),
        ("harmonics_mag_6", _harmonics["harmonics_mag_6"]),
        ("harmonics_mag_7", _harmonics["harmonics_mag_7"]),
        ("harmonics_phase_2", _harmonics["harmonics_phase_2"]),
        ("harmonics_phase_3", _harmonics["harmonics_phase_3"]),
        ("harmonics_phase_4", _harmonics["harmonics_phase_4"]),
        ("harmonics_phase_5", _harmonics["harmonics_phase_5"]),
        ("harmonics_phase_6", _harmonics["harmonics_phase_6"]),
        ("harmonics_phase_7", _harmonics["harmonics_phase_7"]),
        ("harmonics_mse", _harmonics["harmonics_mse"]),
        ("psi_cs", _psi_cs),
        ("psi_eta", _psi_eta),
        # stochastic model features
        ("gp_drw_sigma", _drw_sigma),
        ("gp_drw_tau", _drw_tau),
        ("iar_phi", _iar_phi),
        ("mhps_high", _mhps["mhps_high"]),
        ("mhps_low", _mhps["mhps_low"]),
        ("mhps_non_zero", _mhps["mhps_non_zero"]),
        ("mhps_pn_flag", _mhps["mhps_pn_flag"]),
        ("mhps_ratio", _mhps["mhps_ratio"]),
        ("by_camera", by_camera),
        ("by_field", by_field),
        ("by_band", by_band),
        ("by_camera_field", by_camfld),
        ("cadence_by_camera", cadence_by_camera),
        ("nightly_table", nightly),
        ("seasons", pd.DataFrame(seasons)),
    ])

    return df, summary

def print_summary(summary, max_rows=10):
    def headframe(x, n=max_rows):
        return x.head(n).to_string(index=False)

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
    print(f"std={summary['photometry_std_mag']:.6f}  robust_sigma={summary['photometry_robust_sigma_mag']:.6f}  IQR={summary['photometry_IQR_mag']:.6f}")
    print(f"p05={summary['photometry_p05_mag']:.6f}  p16={summary['photometry_p16_mag']:.6f}  p84={summary['photometry_p84_mag']:.6f}  p95={summary['photometry_p95_mag']:.6f}")
    print(f"clipped mean={summary['clipped_mean_mag_3sigma_about_median']:.6f}  clipped std={summary['clipped_std_mag_3sigma_about_median']:.6f}  outliers={summary['n_outliers_removed_robust_3sigma']}")

    es = summary["error_and_snr_stats"]
    print("\n=== ERRORS / SNR ===")
    print(f"error: mean={es['error_mean']:.6f}  median={es['error_median']:.6f}  p05={es['error_p05']:.6f}  p95={es['error_p95']:.6f}")
    print(f"SNR: median={es['snr_median']:.2f}  p05={es['snr_p05']:.2f}  p95={es['snr_p95']:.2f}")

    print("\n=== VARIABILITY / TREND ===")
    print(f"reduced χ² vs constant={summary['variability_reduced_chi2_vs_constant']:.3f}  | von Neumann={summary['variability_von_neumann_ratio']:.3f}  | lag-1 ρ={summary['variability_lag1_autocorr']:.3f}")
    print(f"Stetson I/J/K={summary['variability_stetson_I']:.3f} / {summary['variability_stetson_J']:.3f} / {summary['variability_stetson_K']:.3f}")
    if "variability_lomb_scargle_best_period_days" in summary:
        print(f"Lomb-Scargle: best_period_days={summary['variability_lomb_scargle_best_period_days']:.6f}  peak_power={summary['variability_lomb_scargle_peak_power']:.6f}  fap={summary['variability_lomb_scargle_fap']:.3e}")
    print(f"trend slope={summary['trend_slope_mag_per_day']:.6e} mag/day ({summary['trend_slope_mag_per_year']:.6e} mag/yr),  R²={summary['trend_r2']:.3f}")

    def _fmt(v, d=4):
        return f"{v:.{d}f}" if np.isfinite(v) else "NaN"

    print("\n=== ALeRCE FEATURES ===")
    print(f"Amplitude={_fmt(summary['amplitude'])}  Beyond1Std={_fmt(summary['beyond_1_std'])}  Con={summary['con']}  delta_mag={_fmt(summary['delta_mag_fid'])}")
    print(f"ExcessVar={_fmt(summary['excess_var'])}  first_mag={_fmt(summary['first_mag'],3)}  Gskew={_fmt(summary['gskew'])}  MaxSlope={_fmt(summary['max_slope'])}")
    print(f"Meanvariance={_fmt(summary['meanvariance'])}  MedianAbsDev={_fmt(summary['median_abs_dev'])}  MedianBRP={_fmt(summary['median_brp'])}  PercentAmplitude={_fmt(summary['percent_amplitude'])}")
    print(f"Q31={_fmt(summary['q31'])}  Skew={_fmt(summary['skew'])}  SmallKurtosis={_fmt(summary['small_kurtosis'])}  Pvar={_fmt(summary['pvar'])}")
    print(f"AndersonDarling={_fmt(summary['anderson_darling'])}  PairSlopeTrend={_fmt(summary['pair_slope_trend'])}  Rcs={_fmt(summary['rcs'])}  Autocor_length={summary['autocor_length']}")
    print(f"SF_ML_amplitude={_fmt(summary['sf_ml_amplitude'])}  SF_ML_gamma={_fmt(summary['sf_ml_gamma'])}")

    print("\n=== HARMONICS (folded LC) ===")
    h_mags = "  ".join(f"H{k}={_fmt(summary[f'harmonics_mag_{k}'])}" for k in range(1, 8))
    print(f"Mag: {h_mags}")
    h_phases = "  ".join(f"φ{k}={_fmt(summary[f'harmonics_phase_{k}'])}" for k in range(2, 8))
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
    print(headframe(summary["by_camera_field"]))

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
            "camera_name",
            "field"]    
    kw = dict(sep=r"\s+", comment="#", engine="python")
    if has_header:
        df = pd.read_csv(path, **kw)
        if len(df.columns) < len(names):
            # pad names if fewer columns
            df.columns = names[:len(df.columns)]
    else:
        df = pd.read_csv(path, header=None, names=names, **kw)

    return df

def main():
    ap = argparse.ArgumentParser(description="Compute rich stats for a photometry .dat file.")
    ap.add_argument("path", help="path to .dat file")
    ap.add_argument("--include-all", action="store_true", help="do NOT filter by good_bad==1 & saturated==0")
    ap.add_argument("--keep-dupes",   action="store_true", help="keep duplicate JD rows instead of dropping")
    ap.add_argument("--has-header",   action="store_true", help="file has a header row")
    ap.add_argument("--lomb-scargle", action="store_true", help="compute Lomb-Scargle periodogram summary stats")
    args = ap.parse_args()

    df = load_dat(args.path, has_header=args.has_header)
    df2, summary = compute_stats(
        df,
        use_only_good=not args.include_all,
        drop_dupes=not args.keep_dupes,
        compute_ls=args.lomb_scargle,
    )
    print_summary(summary)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python stats.py yourfile.dat [--include-all] [--keep-dupes] [--has-header]")
    else:
        main()
