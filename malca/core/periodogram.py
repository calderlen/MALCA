"""Period-finding methods for malca review: LSP, PDM, and Conditional Entropy."""
from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from astropy.timeseries import LombScargle
import numba
import numpy as np

from malca.config import (
    PDM_MIN_PERIOD,
    PDM_MAX_PERIOD,
    PDM_N_PERIODS,
    PDM_N_BINS,
    PDM_PLAVCHAN_COARSE_N_PERIODS,
    PDM_PLAVCHAN_FULL_SCAN_MAX_PERIODS,
    PDM_PLAVCHAN_PHASE_WIDTH,
    PDM_PLAVCHAN_MIN_NEIGHBORS,
    PDM_PLAVCHAN_SHORTLIST_TOP_K,
    PDM_PLAVCHAN_SHORTLIST_WINDOW_STEPS,
    PDM_PLAVCHAN_WORST_FRAC,
    PDM_PLAVCHAN_WORST_MIN,
    PDM_PLAVCHAN_GRID_FACTOR,
    CE_N_PHASE_BINS,
    CE_N_MAG_BINS,
    LS_SAMPLES_PER_PEAK,
    PERIODOGRAM_REFINE_TOP_K,
    PERIODOGRAM_REFINE_WINDOW_STEPS,
    PERIODOGRAM_REFINE_N_GRID,
)


def normalize_pdm_method(method: str | None) -> str:
    """Normalize user-facing PDM method names to internal labels."""
    name = str(method or "classic").strip().lower()
    if name in {"classic", "binned", "theta"}:
        return "classic"
    if name in {"plavchan", "binless", "boxcar", "plavchan_binless"}:
        return "plavchan"
    raise ValueError(f"Unknown PDM method: {method!r}")


def _top_indices(
    metric: np.ndarray,
    *,
    n_top: int,
    maximize: bool,
    min_separation: int = 1,
) -> list[int]:
    """Select top candidate indices with simple index-space separation."""
    arr = np.asarray(metric, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return []

    n_top = max(1, int(n_top))
    min_separation = max(1, int(min_separation))
    order = np.argsort(-arr if maximize else arr)

    chosen: list[int] = []
    for idx in order:
        i = int(idx)
        if not finite[i]:
            continue
        if any(abs(i - j) < min_separation for j in chosen):
            continue
        chosen.append(i)
        if len(chosen) >= n_top:
            break
    return chosen


def _refine_best_period_from_min_metric(
    times: np.ndarray,
    yvals: np.ndarray,
    period_arr: np.ndarray,
    metric_arr: np.ndarray,
    *,
    min_period: float,
    max_period: float,
    n_top: int,
    window_steps: float,
    refine_n_grid: int,
    scan_fn,
    scan_args: tuple,
) -> float:
    """Refine a minimum-metric period by rescanning small windows around top minima."""
    best_idx = int(np.nanargmin(metric_arr))
    best_period = float(period_arr[best_idx])
    best_metric = float(metric_arr[best_idx])

    if period_arr.size < 2:
        return best_period

    refine_n_grid = int(refine_n_grid)
    if refine_n_grid < 3:
        return best_period

    separation = max(1, int(round(abs(window_steps))))
    candidates = _top_indices(
        metric_arr,
        n_top=n_top,
        maximize=False,
        min_separation=separation,
    )
    if not candidates:
        return best_period

    for idx in candidates:
        if period_arr.size < 2:
            continue
        if int(idx) <= 0:
            step = float(abs(period_arr[1] - period_arr[0]))
        elif int(idx) >= period_arr.size - 1:
            step = float(abs(period_arr[-1] - period_arr[-2]))
        else:
            left_step = float(abs(period_arr[int(idx)] - period_arr[int(idx) - 1]))
            right_step = float(abs(period_arr[int(idx) + 1] - period_arr[int(idx)]))
            finite_steps = [val for val in (left_step, right_step) if np.isfinite(val) and val > 0]
            if not finite_steps:
                continue
            step = max(finite_steps)
        if not np.isfinite(step) or step <= 0:
            continue

        half_window = max(step, float(abs(window_steps)) * step)
        center = float(period_arr[int(idx)])
        lo = max(float(min_period), center - half_window)
        hi = min(float(max_period), center + half_window)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue

        local_grid = np.linspace(lo, hi, refine_n_grid)
        local_idx, local_metric = scan_fn(times, yvals, local_grid, *scan_args)
        local_idx = int(local_idx)
        local_best_metric = float(local_metric[local_idx])
        if local_best_metric < best_metric:
            best_metric = local_best_metric
            best_period = float(local_grid[local_idx])

    return best_period


def _build_plavchan_period_grid(
    times: np.ndarray,
    min_period: float,
    max_period: float,
    *,
    grid_factor: float = PDM_PLAVCHAN_GRID_FACTOR,
) -> np.ndarray:
    """Build a Plavchan-style adaptive period grid with dP ~ P^2 / baseline."""
    min_period = float(min_period)
    max_period = float(max_period)
    if not np.isfinite(min_period) or not np.isfinite(max_period):
        return np.asarray([np.nan], dtype=np.float64)
    if max_period < min_period:
        min_period, max_period = max_period, min_period
    if np.isclose(min_period, max_period):
        return np.asarray([min_period], dtype=np.float64)

    times = np.asarray(times, dtype=np.float64)
    finite_times = times[np.isfinite(times)]
    if finite_times.size < 2:
        return np.asarray([min_period, max_period], dtype=np.float64)

    baseline = float(np.max(finite_times) - np.min(finite_times))
    if not np.isfinite(baseline) or baseline <= 0:
        return np.asarray([min_period, max_period], dtype=np.float64)

    factor = float(grid_factor)
    if not np.isfinite(factor) or factor <= 0:
        factor = float(PDM_PLAVCHAN_GRID_FACTOR)

    periods = [min_period]
    current = min_period
    while current < max_period:
        step = factor * current * current / baseline
        if not np.isfinite(step) or step <= 0:
            break
        nxt = current + step
        if nxt >= max_period:
            break
        periods.append(float(nxt))
        current = float(nxt)

    if periods[-1] < max_period:
        periods.append(max_period)
    return np.asarray(periods, dtype=np.float64)


def _downsample_sorted_indices(size: int, *, max_points: int) -> np.ndarray:
    """Return monotonic unique indices spanning a sorted array."""
    size = int(size)
    if size <= 0:
        return np.empty(0, dtype=np.int64)

    max_points = max(1, int(max_points))
    if size <= max_points:
        return np.arange(size, dtype=np.int64)

    idx = np.rint(np.linspace(0, size - 1, num=max_points)).astype(np.int64)
    idx = np.unique(np.clip(idx, 0, size - 1))
    if idx[0] != 0:
        idx = np.concatenate((np.asarray([0], dtype=np.int64), idx))
    if idx[-1] != size - 1:
        idx = np.concatenate((idx, np.asarray([size - 1], dtype=np.int64)))
    return idx


def _plavchan_shortlist_period_grid(
    times: np.ndarray,
    yvals: np.ndarray,
    full_period_arr: np.ndarray,
    *,
    n_bins: int,
    coarse_n_periods: int,
    shortlist_top_k: int,
    shortlist_window_steps: float,
) -> np.ndarray:
    """Use classic PDM and CE on a coarse grid to shortlist Plavchan windows."""
    full_period_arr = np.asarray(full_period_arr, dtype=np.float64)
    if full_period_arr.size <= 1:
        return full_period_arr

    coarse_idx = _downsample_sorted_indices(full_period_arr.size, max_points=coarse_n_periods)
    if coarse_idx.size <= 1:
        return full_period_arr

    coarse_period_arr = full_period_arr[coarse_idx]
    _, classic_metric = _pdm_scan_grid(times, yvals, coarse_period_arr, n_bins)
    _, ce_metric = _ce_scan_grid(times, yvals, coarse_period_arr, CE_N_PHASE_BINS, CE_N_MAG_BINS)

    n_top = max(1, int(shortlist_top_k))
    min_separation = max(1, int(np.ceil(coarse_idx.size / float(max(6 * n_top, 1)))))
    candidate_positions = set(_top_indices(classic_metric, n_top=n_top, maximize=False, min_separation=min_separation))
    candidate_positions.update(_top_indices(ce_metric, n_top=n_top, maximize=False, min_separation=min_separation))
    if not candidate_positions:
        candidate_positions.add(int(np.nanargmin(classic_metric)))
        candidate_positions.add(int(np.nanargmin(ce_metric)))

    shortlist_mask = np.zeros(full_period_arr.size, dtype=bool)
    window_steps = max(0.0, float(shortlist_window_steps))
    for coarse_pos in sorted(candidate_positions):
        coarse_pos = int(np.clip(coarse_pos, 0, coarse_idx.size - 1))
        center_idx = int(coarse_idx[coarse_pos])
        if coarse_pos == 0:
            local_stride = int(coarse_idx[min(1, coarse_idx.size - 1)] - coarse_idx[0])
        elif coarse_pos == coarse_idx.size - 1:
            local_stride = int(coarse_idx[-1] - coarse_idx[-2])
        else:
            local_stride = int(
                max(
                    coarse_idx[coarse_pos] - coarse_idx[coarse_pos - 1],
                    coarse_idx[coarse_pos + 1] - coarse_idx[coarse_pos],
                )
            )
        local_stride = max(local_stride, 1)
        half_window = max(1, int(np.ceil(window_steps * local_stride)))
        lo = max(0, center_idx - half_window)
        hi = min(full_period_arr.size, center_idx + half_window + 1)
        shortlist_mask[lo:hi] = True

    shortlist_mask[0] = True
    shortlist_mask[-1] = True
    shortlist_period_arr = full_period_arr[shortlist_mask]
    return shortlist_period_arr if shortlist_period_arr.size > 0 else full_period_arr


def _refine_lsp_period(
    ls: LombScargle,
    *,
    frequency: np.ndarray,
    power: np.ndarray,
    min_frequency: float,
    max_frequency: float,
    n_top: int,
    window_steps: float,
    refine_n_grid: int,
) -> float:
    """Refine an LSP maximum by rescanning small windows around top peaks."""
    best_idx = int(np.nanargmax(power))
    best_frequency = float(frequency[best_idx])
    best_power = float(power[best_idx])

    if frequency.size < 2:
        return float(1.0 / best_frequency)

    refine_n_grid = int(refine_n_grid)
    if refine_n_grid < 3:
        return float(1.0 / best_frequency)

    dfreq = np.diff(frequency)
    freq_step = float(np.nanmedian(dfreq))
    if not np.isfinite(freq_step) or freq_step <= 0:
        return float(1.0 / best_frequency)

    half_window = max(freq_step, float(abs(window_steps)) * freq_step)
    separation = max(1, int(round(abs(window_steps))))
    candidates = _top_indices(
        power,
        n_top=n_top,
        maximize=True,
        min_separation=separation,
    )
    if not candidates:
        return float(1.0 / best_frequency)

    for idx in candidates:
        center = float(frequency[int(idx)])
        lo = max(float(min_frequency), center - half_window)
        hi = min(float(max_frequency), center + half_window)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            continue

        local_frequency = np.linspace(lo, hi, refine_n_grid)
        local_power = np.asarray(ls.power(local_frequency))
        if local_power.size == 0:
            continue
        local_idx = int(np.argmax(local_power))
        local_best_power = float(local_power[local_idx])
        if local_best_power > best_power:
            best_power = local_best_power
            best_frequency = float(local_frequency[local_idx])

    return float(1.0 / best_frequency)


# ---------------------------------------------------------------------------
# PDM (Phase Dispersion Minimization)
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, cache=True)
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


@numba.jit(nopython=True, cache=True)
def _plavchan_boxcar_smooth_sorted(
    phase_sorted: np.ndarray,
    y_sorted: np.ndarray,
    phase_width: float,
    min_neighbors: int,
) -> np.ndarray:
    """Boxcar-smooth a phase-sorted series with circular wrap."""
    n = len(y_sorted)
    smooth = np.empty(n, dtype=np.float64)
    if n == 0:
        return smooth

    width = phase_width
    if not np.isfinite(width) or width <= 0.0:
        width = 0.0
    if width > 1.0:
        width = 1.0
    half_width = 0.5 * width

    min_neighbors_eff = min_neighbors
    if min_neighbors_eff < 1:
        min_neighbors_eff = 1
    if min_neighbors_eff > n:
        min_neighbors_eff = n

    phase_ext = np.empty(3 * n, dtype=np.float64)
    y_ext = np.empty(3 * n, dtype=np.float64)
    for i in range(n):
        phase_val = phase_sorted[i]
        y_val = y_sorted[i]
        phase_ext[i] = phase_val - 1.0
        phase_ext[n + i] = phase_val
        phase_ext[2 * n + i] = phase_val + 1.0
        y_ext[i] = y_val
        y_ext[n + i] = y_val
        y_ext[2 * n + i] = y_val

    prefix = np.zeros(3 * n + 1, dtype=np.float64)
    for i in range(3 * n):
        prefix[i + 1] = prefix[i] + y_ext[i]

    left = 0
    right = 0
    for i in range(n):
        center_idx = n + i
        center = phase_ext[center_idx]
        left_boundary = center - half_width
        while left < 3 * n and phase_ext[left] < left_boundary:
            left += 1
        if right < left:
            right = left
        right_boundary = center + half_width
        while right < 3 * n and phase_ext[right] <= right_boundary:
            right += 1

        window_left = left
        window_right = right
        count = window_right - window_left
        if count < min_neighbors_eff:
            left_need = (min_neighbors_eff - 1) // 2
            right_need = min_neighbors_eff - 1 - left_need
            window_left = center_idx - left_need
            window_right = center_idx + right_need + 1
            count = window_right - window_left

        smooth[i] = (prefix[window_right] - prefix[window_left]) / float(count)
    return smooth


@numba.jit(nopython=True, cache=True)
def _pdm_theta_plavchan(
    times: np.ndarray,
    yvals: np.ndarray,
    period: float,
    phase_width: float = PDM_PLAVCHAN_PHASE_WIDTH,
    min_neighbors: int = PDM_PLAVCHAN_MIN_NEIGHBORS,
    worst_frac: float = PDM_PLAVCHAN_WORST_FRAC,
    worst_min: int = PDM_PLAVCHAN_WORST_MIN,
) -> float:
    """Compute a Plavchan-style binless PDM statistic for one trial period."""
    n = len(yvals)
    if n < 3:
        return 1.0

    mean_all = 0.0
    for i in range(n):
        mean_all += yvals[i]
    mean_all /= n

    total_ss = 0.0
    for i in range(n):
        diff = yvals[i] - mean_all
        total_ss += diff * diff
    if total_ss <= 0.0 or n < 2:
        return 1.0
    var_total = total_ss / float(n - 1)
    if not np.isfinite(var_total) or var_total <= 0.0:
        return 1.0

    phase = np.mod(times, period) / period
    order = np.argsort(phase)
    phase_sorted = phase[order]
    y_sorted = yvals[order]
    smooth = _plavchan_boxcar_smooth_sorted(
        phase_sorted,
        y_sorted,
        phase_width,
        min_neighbors,
    )

    resid2 = np.empty(n, dtype=np.float64)
    for i in range(n):
        diff = y_sorted[i] - smooth[i]
        resid2[i] = diff * diff

    worst_count = int(np.ceil(float(worst_frac) * n))
    if worst_count < int(worst_min):
        worst_count = int(worst_min)
    if worst_count < 1:
        worst_count = 1
    if worst_count > n:
        worst_count = n

    resid2_sorted = np.sort(resid2)
    worst_sum = 0.0
    for i in range(n - worst_count, n):
        worst_sum += resid2_sorted[i]
    worst_mean = worst_sum / float(worst_count)
    return worst_mean / var_total


@numba.jit(nopython=True, cache=True)
def _pdm_scan_grid(
    times: np.ndarray,
    yvals: np.ndarray,
    period_arr: np.ndarray,
    n_bins: int = PDM_N_BINS,
) -> tuple[int, np.ndarray]:
    """Evaluate PDM theta across a period grid in numba."""
    n_periods = len(period_arr)
    theta = np.empty(n_periods)

    best_idx = 0
    best_theta = np.inf
    for i in range(n_periods):
        theta_i = _pdm_theta(times, yvals, period_arr[i], n_bins)
        theta[i] = theta_i
        if theta_i < best_theta:
            best_theta = theta_i
            best_idx = i

    return best_idx, theta


@numba.jit(nopython=True, cache=True)
def _pdm_scan_grid_plavchan(
    times: np.ndarray,
    yvals: np.ndarray,
    period_arr: np.ndarray,
    phase_width: float = PDM_PLAVCHAN_PHASE_WIDTH,
    min_neighbors: int = PDM_PLAVCHAN_MIN_NEIGHBORS,
    worst_frac: float = PDM_PLAVCHAN_WORST_FRAC,
    worst_min: int = PDM_PLAVCHAN_WORST_MIN,
) -> tuple[int, np.ndarray]:
    """Evaluate Plavchan-style PDM across a period grid in numba."""
    n_periods = len(period_arr)
    theta = np.empty(n_periods)

    best_idx = 0
    best_theta = np.inf
    for i in range(n_periods):
        theta_i = _pdm_theta_plavchan(
            times,
            yvals,
            period_arr[i],
            phase_width,
            min_neighbors,
            worst_frac,
            worst_min,
        )
        theta[i] = theta_i
        if theta_i < best_theta:
            best_theta = theta_i
            best_idx = i

    return best_idx, theta


def pdm_find_period(
    times: np.ndarray,
    yvals: np.ndarray,
    min_period: float = PDM_MIN_PERIOD,
    max_period: float = PDM_MAX_PERIOD,
    n_periods: int = PDM_N_PERIODS,
    n_bins: int = PDM_N_BINS,
    method: str = "classic",
    phase_width: float = PDM_PLAVCHAN_PHASE_WIDTH,
    min_neighbors: int = PDM_PLAVCHAN_MIN_NEIGHBORS,
    refine: bool = False,
    refine_top_k: int = PERIODOGRAM_REFINE_TOP_K,
    refine_window_steps: float = PERIODOGRAM_REFINE_WINDOW_STEPS,
    refine_n_grid: int = PERIODOGRAM_REFINE_N_GRID,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run PDM over a grid of trial periods.

    Returns (best_period, period_array, theta_array).
    Theta is normalized 0-1; lower = better period.
    """
    times = np.asarray(times, dtype=np.float64)
    yvals = np.asarray(yvals, dtype=np.float64)
    t0 = np.min(times)
    t_shifted = times - t0
    method_name = normalize_pdm_method(method)
    if method_name == "plavchan":
        full_period_arr = _build_plavchan_period_grid(
            t_shifted,
            min_period,
            max_period,
            grid_factor=PDM_PLAVCHAN_GRID_FACTOR,
        )
        period_arr = full_period_arr
        if full_period_arr.size > int(PDM_PLAVCHAN_FULL_SCAN_MAX_PERIODS):
            # Use cheap proxy metrics to focus the expensive Plavchan scan on a
            # small set of promising windows for long-baseline searches.
            period_arr = _plavchan_shortlist_period_grid(
                t_shifted,
                yvals,
                full_period_arr,
                n_bins=int(n_bins),
                coarse_n_periods=int(PDM_PLAVCHAN_COARSE_N_PERIODS),
                shortlist_top_k=int(PDM_PLAVCHAN_SHORTLIST_TOP_K),
                shortlist_window_steps=float(PDM_PLAVCHAN_SHORTLIST_WINDOW_STEPS),
            )
        best_idx, theta = _pdm_scan_grid_plavchan(
            t_shifted,
            yvals,
            period_arr,
            float(phase_width),
            int(min_neighbors),
            float(PDM_PLAVCHAN_WORST_FRAC),
            int(PDM_PLAVCHAN_WORST_MIN),
        )
        scan_fn = _pdm_scan_grid_plavchan
        scan_args = (
            float(phase_width),
            int(min_neighbors),
            float(PDM_PLAVCHAN_WORST_FRAC),
            int(PDM_PLAVCHAN_WORST_MIN),
        )
    else:
        period_arr = np.linspace(min_period, max_period, n_periods)
        best_idx, theta = _pdm_scan_grid(t_shifted, yvals, period_arr, n_bins)
        scan_fn = _pdm_scan_grid
        scan_args = (n_bins,)
    best_idx = int(best_idx)
    best_period = float(period_arr[best_idx])
    if bool(refine):
        best_period = _refine_best_period_from_min_metric(
            t_shifted,
            yvals,
            period_arr,
            theta,
            min_period=min_period,
            max_period=max_period,
            n_top=refine_top_k,
            window_steps=refine_window_steps,
            refine_n_grid=refine_n_grid,
            scan_fn=scan_fn,
            scan_args=scan_args,
        )
    return best_period, period_arr, theta


# ---------------------------------------------------------------------------
# Conditional Entropy (CE)
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, cache=True)
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


@numba.jit(nopython=True, cache=True)
def _ce_scan_grid(
    times: np.ndarray,
    yvals: np.ndarray,
    period_arr: np.ndarray,
    n_phase_bins: int = CE_N_PHASE_BINS,
    n_mag_bins: int = CE_N_MAG_BINS,
) -> tuple[int, np.ndarray]:
    """Evaluate conditional entropy across a period grid in numba."""
    n_periods = len(period_arr)
    entropy = np.empty(n_periods)

    best_idx = 0
    best_entropy = np.inf
    for i in range(n_periods):
        entropy_i = _ce_evaluate(times, yvals, period_arr[i], n_phase_bins, n_mag_bins)
        entropy[i] = entropy_i
        if entropy_i < best_entropy:
            best_entropy = entropy_i
            best_idx = i

    return best_idx, entropy


def ce_find_period(
    times: np.ndarray,
    yvals: np.ndarray,
    min_period: float = PDM_MIN_PERIOD,
    max_period: float = PDM_MAX_PERIOD,
    n_periods: int = PDM_N_PERIODS,
    n_phase_bins: int = CE_N_PHASE_BINS,
    n_mag_bins: int = CE_N_MAG_BINS,
    refine: bool = False,
    refine_top_k: int = PERIODOGRAM_REFINE_TOP_K,
    refine_window_steps: float = PERIODOGRAM_REFINE_WINDOW_STEPS,
    refine_n_grid: int = PERIODOGRAM_REFINE_N_GRID,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run Conditional Entropy period search on a uniform-frequency grid.

    Returns (best_period, period_array, entropy_array).
    Lower entropy = better period.
    """
    times = np.asarray(times, dtype=np.float64)
    yvals = np.asarray(yvals, dtype=np.float64)
    t0 = np.min(times)
    t_shifted = times - t0
    min_period = float(min_period)
    max_period = float(max_period)
    if max_period < min_period:
        min_period, max_period = max_period, min_period
    if min_period <= 0 or not np.isfinite(min_period) or not np.isfinite(max_period):
        raise ValueError("CE period bounds must be finite and positive")
    n_periods = max(int(n_periods), 2)
    # Periodogram peak widths are approximately uniform in frequency, not in
    # period.  A linear-period grid badly undersamples short periods when the
    # requested range is wide.  Reverse the reciprocal grid so the public
    # period array remains monotonically increasing for existing consumers.
    frequency_arr = np.linspace(1.0 / max_period, 1.0 / min_period, n_periods)
    period_arr = (1.0 / frequency_arr)[::-1].copy()
    best_idx, entropy = _ce_scan_grid(t_shifted, yvals, period_arr, n_phase_bins, n_mag_bins)
    best_idx = int(best_idx)
    best_period = float(period_arr[best_idx])
    if bool(refine):
        best_period = _refine_best_period_from_min_metric(
            t_shifted,
            yvals,
            period_arr,
            entropy,
            min_period=min_period,
            max_period=max_period,
            n_top=refine_top_k,
            window_steps=refine_window_steps,
            refine_n_grid=refine_n_grid,
            scan_fn=_ce_scan_grid,
            scan_args=(n_phase_bins, n_mag_bins),
        )
    return best_period, period_arr, entropy


# ---------------------------------------------------------------------------
# LSP (Lomb-Scargle Periodogram)
# ---------------------------------------------------------------------------

def lsp_find_period(
    times: np.ndarray,
    yvals: np.ndarray,
    min_period: float = PDM_MIN_PERIOD,
    max_period: float = PDM_MAX_PERIOD,
    samples_per_peak: int = LS_SAMPLES_PER_PEAK,
    refine: bool = False,
    refine_top_k: int = PERIODOGRAM_REFINE_TOP_K,
    refine_window_steps: float = PERIODOGRAM_REFINE_WINDOW_STEPS,
    refine_n_grid: int = PERIODOGRAM_REFINE_N_GRID,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run Lomb-Scargle periodogram via astropy.

    Returns (best_period, period_array, power_array).
    Higher power = better period.
    """


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
    best_period = float(period_arr[best_idx])
    if bool(refine):
        best_period = _refine_lsp_period(
            ls,
            frequency=np.asarray(frequency),
            power=np.asarray(power),
            min_frequency=min_freq,
            max_frequency=max_freq,
            n_top=refine_top_k,
            window_steps=refine_window_steps,
            refine_n_grid=refine_n_grid,
        )
    return best_period, period_arr, np.asarray(power)
