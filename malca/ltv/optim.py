"""
LTV Optimization Utilities.

Advanced optimizations for 17M+ source processing:
- Numba JIT for numerical computations
- Caching layer (joblib) for API results
- Connection pooling for TAP services

"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
from malca.config import LTV_CACHE_DIR
from malca.config import MAD_SCALE

# =============================================================================
# NUMBA JIT FUNCTIONS
# =============================================================================

os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from numba import jit, prange


@jit(nopython=True)
def _detrend_fast(JD: np.ndarray, mag: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Numba-accelerated detrending. 10-100x faster than numpy."""
    n = len(JD)
    jd_min = JD[0]
    result = np.empty(n)
    for i in prange(n):
        t_years = (JD[i] - jd_min) / 365.25
        result[i] = mag[i] - (slope * t_years + intercept)
    return result


@jit(nopython=True)
def _polyfit_linear_fast(x: np.ndarray, y: np.ndarray) -> tuple:
    """Numba-accelerated linear polyfit. Returns (slope, intercept)."""
    n = len(x)
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_xx = 0.0
    
    for i in range(n):
        sum_x += x[i]
        sum_y += y[i]
        sum_xy += x[i] * y[i]
        sum_xx += x[i] * x[i]
    
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        return 0.0, sum_y / n if n > 0 else 0.0
    
    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    
    return slope, intercept


@jit(nopython=True)
def _median_fast(arr: np.ndarray) -> float:
    """Numba-accelerated median."""
    n = len(arr)
    if n == 0:
        return np.nan
    sorted_arr = np.sort(arr)
    if n % 2 == 0:
        return (sorted_arr[n//2 - 1] + sorted_arr[n//2]) / 2.0
    else:
        return sorted_arr[n//2]


@jit(nopython=True)
def _mad_fast(arr: np.ndarray) -> float:
    """Numba-accelerated median absolute deviation."""
    n = len(arr)
    if n == 0:
        return np.nan
    med = _median_fast(arr)
    deviations = np.empty(n)
    for i in range(n):
        deviations[i] = abs(arr[i] - med)
    return _median_fast(deviations) * MAD_SCALE


@jit(nopython=True)
def _season_medians_fast(
    mags: np.ndarray,
    season_idx: np.ndarray,
    min_points: int,
) -> tuple:
    """Numba-accelerated season median computation."""
    # Find unique seasons
    unique_seasons = np.unique(season_idx[season_idx > 0])
    n_seasons = len(unique_seasons)
    
    if n_seasons == 0:
        return np.array([0]), np.array([0.0]), np.array([0.0]), 0
    
    indexes = np.empty(n_seasons, dtype=np.int64)
    medians = np.empty(n_seasons)
    errors = np.empty(n_seasons)
    valid_count = 0
    
    for i, s in enumerate(unique_seasons):
        # Count points in this season
        count = 0
        for j in range(len(season_idx)):
            if season_idx[j] == s:
                count += 1
        
        if count >= min_points:
            # Extract magnitudes for this season
            season_mags = np.empty(count)
            k = 0
            for j in range(len(season_idx)):
                if season_idx[j] == s:
                    season_mags[k] = mags[j]
                    k += 1
            
            indexes[valid_count] = s
            medians[valid_count] = _median_fast(season_mags)
            errors[valid_count] = _mad_fast(season_mags)
            valid_count += 1
    
    return indexes[:valid_count], medians[:valid_count], errors[:valid_count], valid_count

# =============================================================================
# CACHING LAYER
# =============================================================================

from joblib import Memory

# Default cache directory
CACHE_DIR = Path(os.environ.get("LTV_CACHE_DIR", str(LTV_CACHE_DIR)))

_memory = None

def get_cache():
    """Get or create joblib Memory cache."""
    global _memory
    if _memory is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _memory = Memory(str(CACHE_DIR), verbose=0)
    return _memory


def cached(func: Callable) -> Callable:
    """Decorator to cache function results using joblib."""
    return get_cache().cache(func)


def clear_cache():
    """Clear the LTV cache."""
    get_cache().clear()
    print(f"Cleared cache at {CACHE_DIR}")


# =============================================================================
# CONNECTION POOLING
# =============================================================================

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session = None

def get_pooled_session():
    """
    Get a requests Session with connection pooling.
    
    Reuses TCP connections for faster TAP queries.
    Pool size: 20 connections, max 100 total.
    """
    global _session
    if _session is None:
        # Retry strategy
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
        )
        
        # Connection pooling
        adapter = HTTPAdapter(
            pool_connections=20,
            pool_maxsize=100,
            max_retries=retries,
        )
        
        _session = requests.Session()
        _session.mount("http://", adapter)
        _session.mount("https://", adapter)
    
    return _session


def close_session():
    """Close the pooled session."""
    global _session
    if _session is not None:
        _session.close()
        _session = None


# =============================================================================
# UTILITY: Check available optimizations
# =============================================================================

def check_optimizations():
    """Print status of available optimizations."""
    print("LTV Optimization Status:")
    print("  Numba JIT:        ✓ Available")
    print(f"  Joblib cache:     ✓ Available (dir: {CACHE_DIR})")
    print("  Connection pool:  ✓ Available")
