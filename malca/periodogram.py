"""Period-finding methods for malca review: LSP, PDM, and Conditional Entropy."""

from __future__ import annotations

import numba
import numpy as np

from malca.config.config_stats import (
    PDM_MIN_PERIOD,
    PDM_MAX_PERIOD,
    PDM_N_PERIODS,
    PDM_N_BINS,
    CE_N_PHASE_BINS,
    CE_N_MAG_BINS,
    LS_SAMPLES_PER_PEAK,
)


# ---------------------------------------------------------------------------
# PDM (Phase Dispersion Minimization)
# ---------------------------------------------------------------------------

@numba.jit(nopython=True)
def _pdm_theta(times: np.ndarray, yvals: np.ndarray, period: float, n_bins: int = PDM_N_BINS) -> float:
    """Compute PDM theta statistic for a single trial period.

    Theta = (sum of within-bin variances) / (total variance).
    Lower theta means a better period.
    """
    n = len(yvals)
    if n < 3:
        return 1.0

    # Total variance
    mean_all = 0.0
    for i in range(n):
        mean_all += yvals[i]
    mean_all /= n
    var_total = 0.0
    for i in range(n):
        var_total += (yvals[i] - mean_all) ** 2
    var_total /= (n - 1)
    if var_total == 0.0:
        return 1.0

    phase = np.mod(times, period) / period

    # Accumulate per-bin stats in a single pass
    bin_sum = np.zeros(n_bins)
    bin_sum2 = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)
    for i in range(n):
        b = int(phase[i] * n_bins)
        if b >= n_bins:
            b = n_bins - 1
        bin_sum[b] += yvals[i]
        bin_sum2[b] += yvals[i] ** 2
        bin_count[b] += 1

    numerator = 0.0
    denom = 0.0
    for b in range(n_bins):
        nb = bin_count[b]
        if nb > 1:
            mean_b = bin_sum[b] / nb
            var_b = (bin_sum2[b] - nb * mean_b * mean_b) / (nb - 1)
            numerator += (nb - 1) * var_b
            denom += (nb - 1)

    if denom == 0.0:
        return 1.0
    return (numerator / denom) / var_total


def pdm_find_period(
    times: np.ndarray,
    yvals: np.ndarray,
    min_period: float = PDM_MIN_PERIOD,
    max_period: float = PDM_MAX_PERIOD,
    n_periods: int = PDM_N_PERIODS,
    n_bins: int = PDM_N_BINS,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run PDM over a grid of trial periods.

    Returns (best_period, period_array, theta_array).
    Theta is normalized 0-1; lower = better period.
    """
    t0 = np.min(times)
    t_shifted = times - t0
    period_arr = np.linspace(min_period, max_period, n_periods)
    theta = np.empty(n_periods)
    for i in range(n_periods):
        theta[i] = _pdm_theta(t_shifted, yvals, period_arr[i], n_bins=n_bins)
    best_idx = int(np.argmin(theta))
    return float(period_arr[best_idx]), period_arr, theta


# ---------------------------------------------------------------------------
# Conditional Entropy (CE)
# ---------------------------------------------------------------------------

@numba.jit(nopython=True)
def _ce_evaluate(times: np.ndarray, yvals: np.ndarray, period: float,
                 n_phase_bins: int = CE_N_PHASE_BINS, n_mag_bins: int = CE_N_MAG_BINS) -> float:
    """Compute conditional entropy H(mag|phase) for a single trial period.

    Lower entropy = better period (data concentrates in fewer phase-mag cells).
    """
    n = len(yvals)
    if n < 3:
        return 0.0

    phase = np.mod(times, period) / period

    y_min = yvals[0]
    y_max = yvals[0]
    for i in range(1, n):
        if yvals[i] < y_min:
            y_min = yvals[i]
        if yvals[i] > y_max:
            y_max = yvals[i]
    y_range = y_max - y_min
    if y_range == 0.0:
        return 0.0

    # Build 2D histogram
    hist = np.zeros((n_phase_bins, n_mag_bins))
    for i in range(n):
        pb = int(phase[i] * n_phase_bins)
        if pb >= n_phase_bins:
            pb = n_phase_bins - 1
        mb = int((yvals[i] - y_min) / y_range * n_mag_bins)
        if mb >= n_mag_bins:
            mb = n_mag_bins - 1
        hist[pb, mb] += 1.0

    # Normalize to probability
    total = float(n)
    phase_marginal = np.zeros(n_phase_bins)
    for pb in range(n_phase_bins):
        for mb in range(n_mag_bins):
            phase_marginal[pb] += hist[pb, mb]

    # H(mag|phase) = -sum p(phase,mag) * log(p(mag|phase))
    entropy = 0.0
    for pb in range(n_phase_bins):
        if phase_marginal[pb] == 0.0:
            continue
        for mb in range(n_mag_bins):
            if hist[pb, mb] == 0.0:
                continue
            p_joint = hist[pb, mb] / total
            p_cond = hist[pb, mb] / phase_marginal[pb]
            entropy -= p_joint * np.log(p_cond)

    return entropy


def ce_find_period(
    times: np.ndarray,
    yvals: np.ndarray,
    min_period: float = PDM_MIN_PERIOD,
    max_period: float = PDM_MAX_PERIOD,
    n_periods: int = PDM_N_PERIODS,
    n_phase_bins: int = CE_N_PHASE_BINS,
    n_mag_bins: int = CE_N_MAG_BINS,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run Conditional Entropy period search.

    Returns (best_period, period_array, entropy_array).
    Lower entropy = better period.
    """
    t0 = np.min(times)
    t_shifted = times - t0
    period_arr = np.linspace(min_period, max_period, n_periods)
    entropy = np.empty(n_periods)
    for i in range(n_periods):
        entropy[i] = _ce_evaluate(t_shifted, yvals, period_arr[i],
                                  n_phase_bins=n_phase_bins, n_mag_bins=n_mag_bins)
    best_idx = int(np.argmin(entropy))
    return float(period_arr[best_idx]), period_arr, entropy


# ---------------------------------------------------------------------------
# LSP (Lomb-Scargle Periodogram)
# ---------------------------------------------------------------------------

def lsp_find_period(
    times: np.ndarray,
    yvals: np.ndarray,
    min_period: float = PDM_MIN_PERIOD,
    max_period: float = PDM_MAX_PERIOD,
    samples_per_peak: int = LS_SAMPLES_PER_PEAK,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run Lomb-Scargle periodogram via astropy.

    Returns (best_period, period_array, power_array).
    Higher power = better period.
    """
    from astropy.timeseries import LombScargle

    ls = LombScargle(times, yvals)
    min_freq = 1.0 / max_period
    max_freq = 1.0 / min_period
    frequency, power = ls.autopower(
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
        samples_per_peak=samples_per_peak,
    )
    period_arr = 1.0 / frequency
    best_idx = int(np.argmax(power))
    return float(period_arr[best_idx]), period_arr, np.asarray(power)
