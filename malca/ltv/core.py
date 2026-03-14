"""
Refactor of the LTvar-style seasonal-trend code.

Key behavior preserved from the brute-force version:
- Read ASAS-SN .dat light curves, keep only good points and g-band (good/bad==1, v/g?==0)
- Convert times from (jd ~ JD-2450000) to full JD via JD = jd + 2450000
- Special hard-coded Target filter: 17181160895 drops JD < 2.458e6
- Compute “season gap midpoints” from RA and dspring, then keep only midpoints inside [min(JD), max(JD)]
- Require at least 2 midpoints (otherwise skip) like the original (it `continue`s on mid_length==1)
- Cap at 12 seasons by using at most the *earliest 11* midpoints (same net effect as using mid[-1]..mid[-11])
- Define seasons with strict inequalities (points exactly on a midpoint are excluded)
- Compute per-season medians for *non-empty* seasons, keep their original season numbers as x-values
  (this is what the giant if/elif ladder was trying to do with e.g. indexes=[1,5,6,...])
- Fit linear and quadratic polynomials to (season_index, season_median)
- Compute “max diff” using the same (buggy but preserved) algebra as in the snippet.

Notes:
- The original uses ID['ra_deg'] but treats it like hours in the formula.
  Default below preserves that behavior. If your RA is truly degrees, pass --ra-is-deg to convert deg->hours.
"""

from __future__ import annotations

import argparse
import os

# Put this before importing malca.ltv.optim (which uses numba) to prevent 
# thread pool multiplication when using ProcessPoolExecutor
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import re
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.stats import mad_std
from astropy.timeseries import LombScargle
import celerite2
from celerite2 import terms
from scipy.optimize import minimize
from tqdm import tqdm

from malca.config.config_ltv import (
    LTV_DSPRING,
    LTV_MAX_SEASONS,
    LTV_MIN_POINTS_PER_SEASON,
    LTV_MIN_SEASONS_FOR_QUADRATIC,
    LTV_CORE_CHUNK_SIZE,
    LTV_WORKERS,
    LTV_LS_MIN_PERIOD_DAYS,
    LTV_LS_MAX_PERIOD_DAYS,
    LTV_LS_FAP_THRESHOLD,
    LTV_LS_SAMPLES_PER_PEAK,
)
from malca.config.config_paths import LCV2_ROOT, LTV_OUTPUT_DIR
from malca.config.config_pipeline import MAG_BINS, SKYPATROL_JD_OFFSET
from malca.config.config_io import PARQUET_OUTPUT_COMPRESSION
from malca.utils import clean_lc
from malca.stats import inverse_von_neumann_ratio, reduced_chisq, roms_statistic

from malca.ltv.optim import (
    _season_medians_fast,
)


LC_COLUMNS = ["jd", "mag", "error", "good/bad", "camera", "v/g?", "saturated/unsaturated", "camera,field"]

IDX_PATTERN = re.compile(r"lc(\d+)_cal$")


@dataclass(frozen=True)
class Config:
    root: Path
    mag_bin: str
    output: Path
    dspring: float
    ra_is_deg: bool
    max_seasons: int
    min_points_per_season: int
    min_seasons_for_quadratic: int
    write_per_dir: bool
    # Band mode: "pipeline" = use V when available + GP stitch; "g_only" = g-band only, no V, no GP
    band_mode: str
    # Parallel processing options
    workers: int
    chunk_size: int
    overwrite: bool


@dataclass(frozen=True)
class SourceMeta:
    asas_sn_id: int
    ra_deg: float
    dec_deg: float
    pstarrs_g_mag: float


def _build_config(a, mag_bin: str) -> Config:
    """Build a Config for a single mag bin from parsed args."""
    root = Path(a.root)
    out = a.output
    if out is None:
        LTV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = str(LTV_OUTPUT_DIR / f"LTvar{mag_bin.replace('_','-')}.parquet")
    output = Path(out)

    return Config(
        root=root,
        mag_bin=mag_bin,
        output=output,
        dspring=float(a.dspring),
        ra_is_deg=bool(a.ra_is_deg),
        max_seasons=int(a.max_seasons),
        min_points_per_season=int(a.min_points_per_season),
        min_seasons_for_quadratic=int(a.min_seasons_for_quadratic),
        write_per_dir=bool(a.write_per_dir),
        band_mode=str(a.band_mode),
        workers=int(a.workers),
        chunk_size=int(a.chunk_size),
        overwrite=bool(a.overwrite),
    )


def parse_args() -> tuple[list[Config], bool]:
    """Parse CLI args and return (list_of_configs, run_all).

    When ``--all`` is set the list contains one Config per mag bin;
    otherwise it contains a single Config for the requested ``--mag-bin``.
    """
    p = argparse.ArgumentParser(prog="ltv", description="Compute seasonal trends for ASAS-SN light curves.")

    p.add_argument("--root",
                   default=str(LCV2_ROOT),
                   type=str)
    p.add_argument("--mag-bin",
                   default="13_13.5",
                   type=str,
                   choices=[*MAG_BINS, "all"],
                   help=f"Magnitude bin to process (choices: {', '.join(MAG_BINS)}, all)")
    p.add_argument("--output",
                   default=None,
                   type=str,
                   help="Chunked parquet dataset directory (default: LTvar<MAG>.parquet)")
    p.add_argument("--dspring",
                   type=float,
                   default=LTV_DSPRING)
    p.add_argument("--ra-is-deg",
                    action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Convert ID['ra_deg'] from degrees to hours before the dspring formula.")
    p.add_argument("--max-seasons",
                   type=int,
                   default=LTV_MAX_SEASONS)
    p.add_argument("--min-points-per-season",
                   type=int,
                   default=LTV_MIN_POINTS_PER_SEASON,
                   help="Treat seasons with < this many points as empty. (The snippet mostly uses 0, sometimes <=1; default 1 is safest.)",
    )
    p.add_argument("--min-seasons-for-quadratic",
                   type=int,
                   default=LTV_MIN_SEASONS_FOR_QUADRATIC,
                   help="Need at least this many non-empty seasons to do degree-2 polyfit (default 3).",
    )
    p.add_argument("--write-per-dir",
                   action="store_true",
                   help="Write per-directory CSVs to <MAG_BIN>/new/<x>.csv.",
    )
    p.add_argument("--band-mode",
                   type=str,
                   default="pipeline",
                   choices=["pipeline", "g_only"],
                   help="pipeline: use V-band when available and GP stitch; g_only: g-band only, no V-band, no GP correction.",
    )
    p.add_argument("--workers",
                   type=int,
                   default=LTV_WORKERS,
                   help="Number of parallel workers (default: 10)")
    p.add_argument("--chunk-size",
                   type=int,
                   default=LTV_CORE_CHUNK_SIZE,
                   help="Number of results to accumulate before writing (default: 10000)")
    p.add_argument("-o", "--overwrite",
                   action="store_true",
                   help="Start fresh by clearing the checkpoint log instead of resuming prior progress")

    a = p.parse_args()

    run_all = a.mag_bin == "all"

    if run_all:
        if a.output is not None:
            p.error("--output cannot be used with --all (each mag bin auto-resolves its own output path)")
        configs = [_build_config(a, mb) for mb in reversed(MAG_BINS)]
    else:
        configs = [_build_config(a, a.mag_bin)]

    return configs, run_all


def read_index_map(path: Path) -> dict[int, SourceMeta]:
    header = pd.read_csv(path, nrows=0)
    usecols = [c for c in ("asas_sn_id", "ra_deg", "dec_deg", "pstarrs_g_mag") if c in header.columns]
    if "asas_sn_id" not in usecols or "ra_deg" not in usecols or "pstarrs_g_mag" not in usecols:
        raise ValueError(f"Index file missing required columns: {path}")

    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    if "dec_deg" not in df.columns:
        df["dec_deg"] = np.nan

    meta_by_id: dict[int, SourceMeta] = {}
    for row in df.itertuples(index=False):
        target = int(row.asas_sn_id)
        meta_by_id[target] = SourceMeta(
            asas_sn_id=target,
            ra_deg=float(row.ra_deg),
            dec_deg=float(row.dec_deg),
            pstarrs_g_mag=float(row.pstarrs_g_mag),
        )
    return meta_by_id


def iter_light_curve_jobs(
    mag_bin_dir: Path,
    lc_dirs: list[Path],
    processed_files: set[str],
) -> Iterator[tuple[str, SourceMeta]]:
    """Yield light-curve jobs lazily to avoid materializing a whole mag bin."""
    for lc_dir in lc_dirs:
        match = IDX_PATTERN.search(lc_dir.name)
        if match is None:
            continue
        x = int(match.group(1))
        index_path = mag_bin_dir / f"index{x}.csv"

        if not index_path.exists():
            print(f"Skipping lc{x}_cal: missing index{x}.csv")
            continue

        meta_by_id = read_index_map(index_path)

        for file_path in sorted(lc_dir.glob("*.dat2")):
            file_path_str = str(file_path)
            if file_path_str in processed_files:
                continue
            try:
                target = int(file_path.stem)
            except ValueError:
                continue
            meta = meta_by_id.get(target)
            if meta is None:
                continue
            yield file_path_str, meta


def read_lc_dat2_fast(asassn_id: str, path: str, *, include_v: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    dat2_path = os.path.join(path, f"{asassn_id}.dat2")
    if not os.path.exists(dat2_path):
        raise FileNotFoundError(f"Light curve file not found: {dat2_path}")

    columns = ["JD", "mag", "error", "v_g_band", "saturated"]
    df = pd.read_csv(
        dat2_path,
        header=None,
        names=["JD", "mag", "error", "good_bad", "camera#", "v_g_band", "saturated", "cam_field"],
        usecols=columns,
        sep=r"\s+",
        dtype={
            "JD": "float64",
            "mag": "float64",
            "error": "float64",
            "v_g_band": "int8",
            "saturated": "int8",
        },
    )
    df["JD"] = df["JD"] + SKYPATROL_JD_OFFSET

    g_mask = df["v_g_band"] == 0
    df_g = df.loc[g_mask, ["JD", "mag", "error", "saturated"]].reset_index(drop=True)

    if not include_v:
        return df_g, pd.DataFrame(columns=df_g.columns)

    df_v = df.loc[~g_mask, ["JD", "mag", "error", "saturated"]].reset_index(drop=True)
    return df_g, df_v


def filter_lc_for_ltv(df_g: pd.DataFrame, target_id: int) -> pd.DataFrame:
    """Apply clean_lc + special-case target filtering."""
    df = clean_lc(df_g)

    if target_id == 17181160895:
        df = df[df["JD"] >= 2.458e6]

    return df


def compute_vg_overlap_stats(df_g: pd.DataFrame, df_v: pd.DataFrame) -> tuple[bool, float, float]:
    """
    Compute V/g temporal overlap in days and as a fraction of the union.

    Returns: (has_v, overlap_days, overlap_fraction)
    """
    if df_g.empty or "JD" not in df_g.columns:
        return False, 0.0, 0.0

    if df_v.empty or "JD" not in df_v.columns:
        return False, 0.0, 0.0

    g_min = float(df_g["JD"].min())
    g_max = float(df_g["JD"].max())
    v_min = float(df_v["JD"].min())
    v_max = float(df_v["JD"].max())

    overlap = max(0.0, min(g_max, v_max) - max(g_min, v_min))
    union = max(g_max, v_max) - min(g_min, v_min)
    frac = overlap / union if union > 0 else 0.0

    return True, float(overlap), float(frac)


def _fit_joint_band_offset(df_g: pd.DataFrame, df_v: pd.DataFrame, overlap_buffer_days: float = 100.0) -> float:
    """
    Fits a joint Gaussian Process to V and g band data to robustly determine the 
    V-g magnitude offset during their overlap period.
    
    Uses a standard SHO kernel for stellar variability and a custom mean model
    where m(t) = mu + (delta * is_v_band). By optimizing the joint likelihood,
    the GP solves for the offset `delta` that seamlessly aligns the light curves.
    """

    # Determine overlap region with buffer to avoid boundary artifacts
    g_min, g_max = df_g["JD"].min(), df_g["JD"].max()
    v_min, v_max = df_v["JD"].min(), df_v["JD"].max()
    
    overlap_start = max(g_min, v_min) - overlap_buffer_days
    overlap_end = min(g_max, v_max) + overlap_buffer_days
    
    # Filter data to overlap region
    g_mask = (df_g["JD"] >= overlap_start) & (df_g["JD"] <= overlap_end)
    v_mask = (df_v["JD"] >= overlap_start) & (df_v["JD"] <= overlap_end)
    
    g_sub = df_g[g_mask].copy()
    v_sub = df_v[v_mask].copy()
    
    # Combine data for joint fit
    g_sub["is_v"] = 0.0
    v_sub["is_v"] = 1.0
    
    df_joint = pd.concat([g_sub, v_sub], ignore_index=True).sort_values("JD").reset_index(drop=True)
    
    if len(df_joint) < 10:
        # Fallback to simple median matching if not enough points for GP
        return float(np.median(v_sub["mag"]) - np.median(g_sub["mag"]))

    t = df_joint["JD"].values
    y = df_joint["mag"].values
    yerr = df_joint["error"].values if "error" in df_joint.columns else np.full_like(y, 0.02)
    is_v_flag = df_joint["is_v"].values

    # Center timestamps for numerical stability
    t_center = np.median(t)
    t_obj = t - t_center

    # Custom celerite2 mean model: mu + (is_v * delta)
    class BandOffsetMeanModel(celerite2.MeanModel):
        def __init__(self, mu, delta, is_v_flag):
            self.mu = mu
            self.delta = delta
            self.is_v_flag = is_v_flag
            self.parameter_names = ("mu", "delta")
            
        def get_value(self, x):
            return self.mu + self.is_v_flag * self.delta
            
        def compute_gradient(self, x):
            return np.array([np.ones_like(x), self.is_v_flag])
            
    # Initial guesses
    mu_guess = np.median(g_sub["mag"]) if len(g_sub) > 0 else np.median(y)
    v_med = np.median(v_sub["mag"]) if len(v_sub) > 0 else mu_guess
    delta_guess = v_med - mu_guess

    mean_model = BandOffsetMeanModel(mu=mu_guess, delta=delta_guess, is_v_flag=is_v_flag)
    
    # Kernel: SHOTerm (damped random walk/stellar variability) + Jitter
    sigma_guess = np.var(y)
    rho_guess = 100.0 # days
    kernel = terms.SHOTerm(sigma=sigma_guess, rho=rho_guess, Q=1.0) + terms.JitterTerm(log_sigma=np.log(np.median(yerr)))

    gp = celerite2.GaussianProcess(kernel, mean=mean_model)
    gp.compute(t_obj, yerr=yerr)

    def neg_log_like(params):
        gp.set_parameter_vector(params)
        try:
            return -gp.log_likelihood(y)
        except Exception:
            return 1e10

    initial_params = gp.get_parameter_vector()
    
    # Bounds to prevent unphysical regimes
    bounds = gp.get_parameter_bounds()
    # Relax bounds slightly for the mean model parameters
    bounds[-2] = (None, None) # mu
    bounds[-1] = (None, None) # delta

    try:
        soln = minimize(neg_log_like, initial_params, method="L-BFGS-B", bounds=bounds)
        gp.set_parameter_vector(soln.x)
        final_delta = float(gp.mean.delta)
    except Exception:
        # Fallback if optimization fails
        final_delta = float(delta_guess)

    return final_delta


def stitch_bands_celerite(df_g: pd.DataFrame, df_v: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """
    Computes V-g offset via joint GP in overlap region, subtracts it from V-band,
    and returns a seamlessly integrated light curve.
    
    Returns:
        df_combined (pd.DataFrame): Sorted, stitched combined light curve.
        vg_offset_applied (float): The actual Delta-mag offset subtracted from V.
    """
    offset = _fit_joint_band_offset(df_g, df_v)
    
    # Apply global magnitude shift to ALL V-band data
    df_v_shifted = df_v.copy()
    df_v_shifted["mag"] -= offset
    
    # Combine and sort by time
    df_combined = pd.concat([df_g, df_v_shifted], ignore_index=True)
    df_combined = df_combined.sort_values("JD").reset_index(drop=True)
    
    return df_combined, offset


def seasonal_midpoints_from_ra(
    ra_val: float, *,
    ra_is_deg: bool,
    dspring: float,
    n_midpoints: int,
) -> np.ndarray:
    """
    Replicates:
      date1 = dspring + 365.25*(RA-12.0)/24.0
      date2 = date1 + 365.25/2.0 + 365.25
      mid(n) = date2 - n*365.25
    """
    ra_hours = ra_val / 15.0 if ra_is_deg else ra_val
    n = np.arange(n_midpoints, dtype=float)

    date1 = dspring + 365.25 * (ra_hours - 12.0) / 24.0
    date2 = date1 + 365.25 / 2.0 + 365.25
    mid = date2 - n * 365.25

    return np.asarray(mid, dtype=float)


def choose_midpoints_in_range(mid: np.ndarray, tmin: float, tmax: float, max_seasons: int) -> np.ndarray:
    """
    Keep only midpoints within (tmin, tmax) like the snippet, then cap to max_seasons.

    The brute-force code effectively uses at most 11 midpoints (for 12 seasons) by indexing from the end.
    Since mid is generated in decreasing order, mid[-1]..mid[-11] are the 11 *smallest* midpoints.
    Equivalent, once sorted ascending: keep the earliest 11 midpoints (smallest times).

    Returns sorted ascending midpoints.
    """
    mid_in = mid[(mid > tmin) & (mid < tmax)]
    if mid_in.size == 0:
        return np.array([], dtype=float)

    mid_sorted = np.sort(mid_in)

    max_midpoints = max_seasons - 1
    if mid_sorted.size > max_midpoints:
        mid_sorted = mid_sorted[:max_midpoints]

    return mid_sorted


def assign_seasons_strict(JD: np.ndarray, mids_sorted: np.ndarray) -> np.ndarray:
    """
    Seasons are defined by strict inequalities:

      S=1: JD < mid0
      S=2: mid0 < JD < mid1
      ...
      S=k: mid_{k-2} < JD < mid_{k-1}
      S=k+1: JD > mid_{k-1}

    Points exactly equal to any midpoint are excluded (strict).
    """
    if mids_sorted.size == 0:
        return np.full(JD.shape, -1, dtype=int)

    # Exclude exact-boundary points (strict inequalities)
    mask = np.ones(JD.shape, dtype=bool)
    for m in mids_sorted:
        mask &= ~np.isclose(JD, m, rtol=0.0, atol=0.0)

    JD2 = JD[mask]
    # np.digitize: returns 0..len(mids), where 0 means < mids[0], len(mids) means > mids[-1]
    season2 = np.digitize(JD2, mids_sorted, right=False) + 1  # seasons numbered starting at 1

    season = np.full(JD.shape, -1, dtype=int)
    season[mask] = season2
    return season


def season_medians_with_gap_indices(
    mags: np.ndarray,
    season_idx: np.ndarray,
    *,
    min_points_per_season: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute median and MAD per season for seasons that have enough points.
    Return (indexes, meds, meds_err) where indexes are the *season numbers* (gap-preserving).
    """
    good = season_idx > 0
    if not np.any(good):
        return np.array([]), np.array([]), np.array([])

    indexes, meds, errs, count = _season_medians_fast(
        mags.astype(np.float64),
        season_idx.astype(np.int64),
        min_points_per_season,
    )
    if count == 0:
        return np.array([]), np.array([]), np.array([])
    order = np.argsort(indexes)
    return indexes[order], meds[order], errs[order]


def compute_trend_metrics(indexes: np.ndarray, meds: np.ndarray) -> tuple[float, float, float, float, float]:
    """
    Return:
      (lin_slope, quad_slope, coeff1, coeff2, max_diff)

    Preserves the snippet's naming/buggy algebra:
      coeffs = polyfit(x, y, 2) gives [a,b,c]
      quadratic_slope = a
      c1 = b
      c2 = c
      te = -c2/(2*a)   (bug preserved)
      me = c1-(c2^2)/(4*a) (bug preserved)
      m(x) = c1 + c2*x + a*x^2 (bug preserved)
    """
    # Linear slope
    lin = np.polyfit(indexes, meds, 1)
    lin_slope = float(lin[0])

    # Quadratic
    quad = np.polyfit(indexes, meds, 2)
    a = float(quad[0])
    c1 = float(quad[1])
    c2 = float(quad[2])

    # Fitted endpoints (preserved)
    x0 = float(indexes[0])
    x1 = float(indexes[-1])

    m0 = c1 + c2 * x0 + a * x0 * x0
    m1 = c1 + c2 * x1 + a * x1 * x1

    # Handle near-linear case safely
    if np.isclose(a, 0.0):
        diff = abs(m1 - m0)
        return lin_slope, a, c1, c2, float(diff)

    te = -c2 / (2.0 * a)
    me = c1 - (c2 * c2) / (4.0 * a)

    if (te > x0) and (te < x1):
        m1m0 = abs(m1 - m0)
        m1me = abs(m1 - me)
        m0me = abs(m0 - me)
        diff = max(m1m0, m1me, m0me)
    else:
        diff = abs(m1 - m0)

    return lin_slope, a, c1, c2, float(diff)


def compute_basic_lc_stats(JD: np.ndarray) -> dict[str, float | int]:
    """Compute cheap whole-light-curve coverage stats."""
    if JD.size == 0:
        return {
            "n_points": 0,
            "time_span_days": np.nan,
            "n_unique_nights": 0,
        }

    return {
        "n_points": int(JD.size),
        "time_span_days": float(JD.max() - JD.min()),
        "n_unique_nights": int(np.unique(np.floor(JD)).size),
    }


def compute_season_diagnostics(
    meds: np.ndarray,
    season_times: np.ndarray,
    season_spans: np.ndarray,
    season_counts: np.ndarray,
) -> dict[str, float | int]:
    """Compute season-level robustness diagnostics."""
    out: dict[str, float | int] = {
        "n_seasons": int(meds.size),
        "season_points_min": np.nan,
        "season_points_median": np.nan,
        "season_points_max": np.nan,
        "season_span_days_mean": np.nan,
        "season_span_days_median": np.nan,
        "season_span_days_max": np.nan,
        "season_step_max_mag": np.nan,
        "season_step_mean_abs_mag": np.nan,
        "season_step_max_fraction": np.nan,
        "season_monotonicity_fraction": np.nan,
        "season_spearman_rho": np.nan,
        "season_kendall_tau": np.nan,
        "leave1out_slope_std": np.nan,
        "leave1out_slope_range": np.nan,
    }

    if season_counts.size > 0:
        out["season_points_min"] = int(np.min(season_counts))
        out["season_points_median"] = float(np.median(season_counts))
        out["season_points_max"] = int(np.max(season_counts))

    if season_spans.size > 0:
        out["season_span_days_mean"] = float(np.mean(season_spans))
        out["season_span_days_median"] = float(np.median(season_spans))
        out["season_span_days_max"] = float(np.max(season_spans))

    if meds.size >= 2:
        steps = np.diff(meds)
        abs_steps = np.abs(steps)
        total_change = float(abs(meds[-1] - meds[0]))
        out["season_step_max_mag"] = float(np.max(abs_steps))
        out["season_step_mean_abs_mag"] = float(np.mean(abs_steps))
        out["season_step_max_fraction"] = float(np.max(abs_steps) / max(total_change, 1e-6))
        if total_change > 0:
            out["season_monotonicity_fraction"] = float(np.mean((steps * (meds[-1] - meds[0])) >= 0.0))

    if season_times.size >= 2:
        try:
            rho = pd.Series(season_times).corr(pd.Series(meds), method="spearman")
            out["season_spearman_rho"] = float(rho)
        except Exception:
            pass
        try:
            tau = pd.Series(season_times).corr(pd.Series(meds), method="kendall")
            out["season_kendall_tau"] = float(tau)
        except Exception:
            pass

    if season_times.size >= 3:
        loo_slopes = []
        for i in range(season_times.size):
            mask = np.ones(season_times.size, dtype=bool)
            mask[i] = False
            x = season_times[mask]
            y = meds[mask]
            if x.size < 2 or np.allclose(x, x[0]):
                continue
            try:
                loo_slopes.append(float(np.polyfit(x, y, 1)[0]))
            except Exception:
                continue
        if loo_slopes:
            loo_arr = np.asarray(loo_slopes, dtype=float)
            out["leave1out_slope_std"] = float(np.std(loo_arr, ddof=0))
            out["leave1out_slope_range"] = float(np.max(loo_arr) - np.min(loo_arr))

    return out


def _bic_from_residuals(resid: np.ndarray, n_params: int) -> float:
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    n = resid.size
    if n <= n_params:
        return np.nan
    rss = float(np.sum(resid * resid))
    return float(n * np.log(max(rss / n, 1e-12)) + n_params * np.log(n))


def compute_time_trend_diagnostics(t_years: np.ndarray, meds: np.ndarray) -> dict[str, float]:
    """Compute time-based trend diagnostics from seasonal medians."""
    out = {
        "trend_slope_mag_per_year": np.nan,
        "trend_quad_mag_per_year2": np.nan,
        "trend_slope_err_mag_per_year": np.nan,
        "trend_slope_snr": np.nan,
        "trend_r2": np.nan,
        "trend_delta_bic_linear": np.nan,
        "trend_delta_bic_quadratic": np.nan,
    }

    x = np.asarray(t_years, dtype=float)
    y = np.asarray(meds, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 2:
        return out

    x = x[mask]
    y = y[mask]
    if np.allclose(x, x[0]):
        return out

    lin = np.polyfit(x, y, 1)
    yhat_lin = np.polyval(lin, x)
    resid_lin = y - yhat_lin
    ss_res = float(np.sum(resid_lin * resid_lin))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    out["trend_slope_mag_per_year"] = float(lin[0])
    out["trend_r2"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    sxx = float(np.sum((x - np.mean(x)) ** 2))
    if x.size > 2 and sxx > 0:
        sigma2 = ss_res / (x.size - 2)
        if np.isfinite(sigma2) and sigma2 >= 0:
            slope_err = float(np.sqrt(sigma2 / sxx))
            out["trend_slope_err_mag_per_year"] = slope_err
            if slope_err > 0:
                out["trend_slope_snr"] = float(lin[0] / slope_err)

    bic_const = _bic_from_residuals(y - np.mean(y), 1)
    bic_lin = _bic_from_residuals(resid_lin, 2)
    if np.isfinite(bic_const) and np.isfinite(bic_lin):
        out["trend_delta_bic_linear"] = float(bic_const - bic_lin)

    if x.size >= 3:
        quad = np.polyfit(x, y, 2)
        yhat_quad = np.polyval(quad, x)
        bic_quad = _bic_from_residuals(y - yhat_quad, 3)
        out["trend_quad_mag_per_year2"] = float(quad[0])
        if np.isfinite(bic_lin) and np.isfinite(bic_quad):
            out["trend_delta_bic_quadratic"] = float(bic_lin - bic_quad)

    return out


def compute_lomb_scargle(
    JD: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray | None = None,
    *,
    lin_coeff: tuple[float, float] | None = None,
    quad_coeff: tuple[float, float, float] | None = None,
    detrend_mode: str = "linear",
    intercept: float | None = None,
    min_period_days: float = LTV_LS_MIN_PERIOD_DAYS,
    max_period_days: float = LTV_LS_MAX_PERIOD_DAYS,
    fap_threshold: float | None = None,
    samples_per_peak: int = LTV_LS_SAMPLES_PER_PEAK,
) -> dict:
    """
    Compute Lomb-Scargle periodogram on detrended light curve.
    
    Paper method:
    - Subtract linear/quadratic trend from light curve
    - Use Lomb-Scargle to search for periods > 10 days
    - Discard periods longer than observing season (set via max_period_days)
    - Report best period, power, and FAP
    
    Returns dict with:
    - ls_period: Best period in days (if significant)
    - ls_power: Power at best period
    - ls_fap: False alarm probability
    """
    result = {
        "ls_period": np.nan,
        "ls_power": np.nan,
        "ls_fap": np.nan,
    }
    
    if len(JD) < 50:
        return result
    
    # Detrend using linear or quadratic fit
    t_years = (JD - JD.min()) / 365.25
    if detrend_mode == "quadratic" and quad_coeff is not None:
        a, b, c = quad_coeff
        trend = a * t_years * t_years + b * t_years + c
    elif lin_coeff is not None:
        m, b = lin_coeff
        trend = m * t_years + b
    else:
        baseline = float(np.median(mag)) if intercept is None else float(intercept)
        trend = np.full_like(mag, baseline, dtype=float)

    mag_detrended = mag - trend
    
    # Compute Lomb-Scargle
    if err is not None and len(err) == len(JD):
        ls = LombScargle(JD, mag_detrended, err)
    else:
        ls = LombScargle(JD, mag_detrended)
    
    # Frequency grid: periods from min_period to max_period
    if max_period_days <= 0 or max_period_days < min_period_days:
        return result
    min_freq = 1.0 / max_period_days
    max_freq = 1.0 / min_period_days
    
    try:
        freq, power = ls.autopower(
            minimum_frequency=min_freq,
            maximum_frequency=max_freq,
            samples_per_peak=samples_per_peak,
        )
        
        if len(power) == 0:
            return result
        
        # Best period
        best_idx = np.argmax(power)
        best_power = float(power[best_idx])
        best_period = float(1.0 / freq[best_idx])
        
        # False alarm probability
        fap = float(ls.false_alarm_probability(best_power))
        
        if fap_threshold is None or fap <= fap_threshold:
            result["ls_period"] = best_period
            result["ls_power"] = best_power
        result["ls_fap"] = fap
        
    except Exception:
        pass
    
    return result


def process_one_lc(
    path: str,
    meta: SourceMeta,
    cfg: Config,
) -> dict | None:
    basename = os.path.basename(path)
    asassn_id = basename.replace('.dat2', '')
    target = meta.asas_sn_id
    ra_val = meta.ra_deg
    p_mag = meta.pstarrs_g_mag

    dir_path = os.path.dirname(path)
    df_g, df_v = read_lc_dat2_fast(asassn_id, dir_path, include_v=(cfg.band_mode != "g_only"))

    if df_g.empty:
        return None

    df = filter_lc_for_ltv(df_g, target)

    if df.empty:
        return None

    # V/g overlap statistics retained as diagnostic metadata
    df_v_clean = clean_lc(df_v) if not df_v.empty else df_v
    vg_has_v, vg_overlap_days, vg_overlap_frac = compute_vg_overlap_stats(df, df_v_clean)
    
    vg_offset_applied = np.nan
    # Perform Joint GP Stitching if we have usable V-band overlap (e.g. at least 10 days)
    if vg_has_v and vg_overlap_days > 10.0:
        try:
            df, vg_offset_applied = stitch_bands_celerite(df, df_v_clean)
        except Exception:
            pass # Fallback to using just g-band if stitching catastrophically fails

    JD = df["JD"].to_numpy(dtype=float)
    mag = df["mag"].to_numpy(dtype=float)
    err = df["error"].to_numpy(dtype=float) if "error" in df.columns else np.full(mag.shape[0], np.nan)
    lc_median = float(np.median(mag))
    lc_mad = float(mad_std(mag))
    lc_dispersion = float(np.ptp(mag))
    vnr = float(inverse_von_neumann_ratio(mag))
    rchisq = float(reduced_chisq(mag, err, lc_median))
    roms = float(roms_statistic(mag, err))
    basic_stats = compute_basic_lc_stats(JD)

    mid_all = seasonal_midpoints_from_ra(
        ra_val,
        ra_is_deg=cfg.ra_is_deg,
        dspring=cfg.dspring,
        n_midpoints=cfg.max_seasons,
    )
    mids = choose_midpoints_in_range(mid_all, float(JD.min()), float(JD.max()), cfg.max_seasons)

    if mids.size < 2:
        return None

    season_idx = assign_seasons_strict(JD, mids)
    indexes, meds, _meds_err = season_medians_with_gap_indices(
        mag, season_idx, min_points_per_season=cfg.min_points_per_season
    )

    if meds.size < cfg.min_seasons_for_quadratic:
        return None

    lin_slope, quad_slope, c1, c2, diff = compute_trend_metrics(indexes.astype(float), meds.astype(float))

    # Fit seasonal medians vs time for LS detrending
    season_times = []
    season_spans = []
    season_counts = []
    for s in indexes:
        sel = season_idx == s
        if not np.any(sel):
            continue
        season_times.append(np.median(JD[sel]))
        season_spans.append(JD[sel].max() - JD[sel].min())
        season_counts.append(int(np.count_nonzero(sel)))

    season_times = np.asarray(season_times, dtype=float)
    season_spans = np.asarray(season_spans, dtype=float)
    season_counts = np.asarray(season_counts, dtype=int)
    t_years = (season_times - JD.min()) / 365.25 if season_times.size > 0 else np.array([])
    season_stats = compute_season_diagnostics(meds, t_years, season_spans, season_counts)
    trend_stats = compute_time_trend_diagnostics(t_years, meds)

    lin_coeff = None
    quad_coeff = None
    if t_years.size >= 2:
        lin_coeff = tuple(np.polyfit(t_years, meds, 1))
    if t_years.size >= cfg.min_seasons_for_quadratic:
        quad_coeff = tuple(np.polyfit(t_years, meds, 2))

    avg_season_span_days = float(np.mean(season_spans)) if season_spans.size > 0 else None

    # Compute Lomb-Scargle on detrended light curve (paper: periods > 10 days)
    err_ls = err if np.any(np.isfinite(err) & (err > 0)) else None
    detrend_mode = "quadratic" if quad_coeff is not None else "linear"
    max_period_days = avg_season_span_days if avg_season_span_days is not None else LTV_LS_MAX_PERIOD_DAYS
    ls_result = compute_lomb_scargle(
        JD, mag, err_ls,
        lin_coeff=lin_coeff,
        quad_coeff=quad_coeff,
        detrend_mode=detrend_mode,
        intercept=lc_median,
        min_period_days=LTV_LS_MIN_PERIOD_DAYS,
        max_period_days=max_period_days,
        fap_threshold=LTV_LS_FAP_THRESHOLD,
    )

    return {
        "ASAS-SN ID": target,
        "ra_deg": ra_val,
        "dec_deg": meta.dec_deg,
        "Pstarss gmag": p_mag,
        **basic_stats,
        "Median": lc_median,
        "Median_err": lc_mad,
        "Dispersion": lc_dispersion,
        "Slope": lin_slope,
        "Quad Slope": quad_slope,
        "coeff1": c1,
        "coeff2": c2,
        "max diff": diff,
        **season_stats,
        **trend_stats,
        "vg_has_v": vg_has_v,
        "vg_overlap_days": vg_overlap_days,
        "vg_overlap_fraction": vg_overlap_frac,
        "vg_offset_applied": vg_offset_applied,
        "ls_period": ls_result["ls_period"],
        "ls_power": ls_result["ls_power"],
        "ls_fap": ls_result["ls_fap"],
        "inverse_von_neumann_ratio": vnr,
        "reduced_chi2_vs_constant": rchisq,
        "roms": roms,
        "lc_path": path,
    }

class ChunkedParquetWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.path.glob("chunk_*.parquet"))
        if existing:
            try:
                last = existing[-1].stem.split("_")[-1]
                self.counter = int(last) + 1
            except Exception:
                self.counter = len(existing)
        else:
            self.counter = 0

    def write_chunk(self, chunk_results):
        if not chunk_results:
            return
        df_chunk = pd.DataFrame(chunk_results)
        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
        tmp_path = self.path / f"chunk_{self.counter:06d}.parquet.tmp"
        final_path = self.path / f"chunk_{self.counter:06d}.parquet"
        pq.write_table(table, tmp_path, compression=PARQUET_OUTPUT_COMPRESSION)
        os.replace(tmp_path, final_path)
        self.counter += 1

    def close(self):
        return


def make_writer(path: Path | None):
    if path is None:
        return None
    return ChunkedParquetWriter(path)


def run_mag_bin(cfg: Config) -> None:
    """Run the full LTV pipeline for a single magnitude bin."""
    mag_bin_dir = cfg.root / cfg.mag_bin
    lc_dirs = sorted(mag_bin_dir.glob("lc*_cal"))
    lc_dirs = [d for d in lc_dirs if d.is_dir() and IDX_PATTERN.search(d.name)]

    print(f"Processing mag_bin={cfg.mag_bin}: found {len(lc_dirs)} lc_cal directories")
    print(f"Workers: {cfg.workers}, Chunk size: {cfg.chunk_size}, Output: chunked parquet dataset")

    output_path = Path(cfg.output)
    checkpoint_log = output_path.with_name(f"{output_path.stem}_PROCESSED.txt")

    processed_files = set()
    if checkpoint_log.exists() and cfg.overwrite:
        try:
            with open(checkpoint_log, "w"):
                pass
            print(f"Overwriting checkpoint log: {checkpoint_log}")
        except Exception as e:
            print(f"Warning: could not overwrite checkpoint log {checkpoint_log}: {e}")

    if checkpoint_log.exists() and not cfg.overwrite:
        print(f"Resume mode: loading checkpoint from {checkpoint_log}")
        with open(checkpoint_log, "r") as f:
            processed_files = set(line.strip() for line in f)
        print(f"Found {len(processed_files)} previously processed files")

    max_in_flight = max(1, cfg.workers * 2)
    job_iter = iter_light_curve_jobs(mag_bin_dir, lc_dirs, processed_files)

    with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
        pending = {}
        exhausted = False
        submitted = 0

        def submit_next_job() -> bool:
            nonlocal exhausted, submitted
            if exhausted:
                return False
            try:
                file_path, meta = next(job_iter)
            except StopIteration:
                exhausted = True
                return False
            future = executor.submit(process_one_lc, file_path, meta, cfg)
            pending[future] = file_path
            submitted += 1
            return True

        while len(pending) < max_in_flight and submit_next_job():
            pass

        if submitted == 0:
            print("No files to process (all may be completed)")
            return

        writer = make_writer(output_path)
        if writer is None:
            raise ValueError(f"Could not create writer for output path: {output_path}")
        results = []
        total_written = 0

        def write_chunk(chunk_results):
            nonlocal total_written, results
            if not chunk_results:
                return

            writer.write_chunk(chunk_results)

            with open(checkpoint_log, "a") as f:
                for row in chunk_results:
                    f.write(row.get('_path', '') + "\n")

            total_written += len(chunk_results)
            print(f"Wrote chunk: {len(chunk_results)} rows (total: {total_written})")

        print(f"Processing light curve files with at most {max_in_flight} in flight")

        with tqdm(total=submitted, desc="Processing LCs", unit="lc") as pbar:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    file_path = pending.pop(future)
                    pbar.update(1)
                    try:
                        result = future.result()
                        if result is not None:
                            result['_path'] = file_path
                            results.append(result)

                            if len(results) >= cfg.chunk_size:
                                write_chunk(results)
                                results = []
                    except Exception as e:
                        print(f"ERROR processing {file_path}: {e}")

                    while len(pending) < max_in_flight and submit_next_job():
                        pbar.total = submitted
                        pbar.refresh()

    if results:
        write_chunk(results)

    if writer:
        writer.close()

    print(f"Complete! Wrote {total_written} rows to {output_path}")


def main() -> None:
    configs, run_all = parse_args()

    if run_all:
        for i, cfg in enumerate(configs, 1):
            print(f"\n{'='*60}")
            print(f"  Mag bin {i}/{len(configs)}: {cfg.mag_bin}")
            print(f"{'='*60}")
            run_mag_bin(cfg)
        print(f"\nAll {len(configs)} mag bins complete!")
    else:
        run_mag_bin(configs[0])


if __name__ == "__main__":
    main()
