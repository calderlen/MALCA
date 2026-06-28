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
- Compute "max diff" from the correctly evaluated fitted trend over the observed season range.

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
from astropy.stats import bayesian_blocks, mad_std
from astropy.timeseries import LombScargle
import celerite2
from celerite2 import terms
from scipy.optimize import minimize
from scipy import stats as sp_stats
from tqdm import tqdm

from malca.config import (
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
    LTV_SMOOTH_WINDOW_DAYS,
    LTV_SMOOTH_WINDOWS_DAYS,
    LTV_SMOOTH_MIN_POINTS,
    LTV_LOWESS_FRAC,
    LTV_LOWESS_ROBUST_ITERS,
    LTV_BINNED_SF_BIN_DAYS,
    LTV_BINNED_SF_LAG_BINS_DAYS,
)
from malca.config import MALCA_LCV2_ROOT_ENV, require_lcv2_root
from malca.config import MAG_BINS, SKYPATROL_JD_OFFSET
from malca.config import PARQUET_OUTPUT_COMPRESSION
from malca.products.feature_layers import to_layer_first_frame
from malca.core.utils import clean_lc
from malca.core.stats import inverse_von_neumann_ratio, reduced_chisq, roms_statistic
from malca.ltv.paths import DEFAULT_LTV_RUN_DIR, ltv_core_output_path

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
    # File extension for light curve files
    file_ext: str


@dataclass(frozen=True)
class SourceMeta:
    asas_sn_id: int
    ra_deg: float
    dec_deg: float
    pstarrs_g_mag: float


def _build_config(a, mag_bin: str) -> Config:
    """Build a Config for a single mag bin from parsed args."""
    from malca.config import LTV_LIGHT_CURVE_FILE_EXTENSION

    root = require_lcv2_root(a.root)
    out = a.output
    if out is None:
        out = str(ltv_core_output_path(mag_bin, a.run_dir))
    output = Path(out)

    file_ext = a.extension if a.extension is not None else LTV_LIGHT_CURVE_FILE_EXTENSION

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
        file_ext=file_ext,
    )


def parse_args() -> tuple[list[Config], bool]:
    """Parse CLI args and return (list_of_configs, run_all).

    When ``--all`` is set the list contains one Config per mag bin;
    otherwise it contains a single Config for the requested ``--mag-bin``.
    """
    p = argparse.ArgumentParser(prog="ltv", description="Compute seasonal trends for ASAS-SN light curves.")

    p.add_argument("--root",
                   default=None,
                   type=str,
                   help=f"Raw ASAS-SN light-curve root; defaults to ${MALCA_LCV2_ROOT_ENV}")
    p.add_argument("--run-dir",
                   default=str(DEFAULT_LTV_RUN_DIR),
                   type=str,
                   help=f"LTV run directory for default outputs (default: {DEFAULT_LTV_RUN_DIR})")
    p.add_argument("--mag-bin",
                   default="13_13.5",
                   type=str,
                   choices=[*MAG_BINS, "all"],
                   help=f"Magnitude bin to process (choices: {', '.join(MAG_BINS)}, all)")
    p.add_argument("--output",
                   default=None,
                   type=str,
                   help="Chunked parquet dataset directory (default: <run-dir>/results/LTvar<MAG>.parquet)")
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
                   help="Deprecated no-op; outputs are written as Parquet.",
    )
    p.add_argument("--band-mode",
                   type=str,
                   default="pipeline",
                   choices=["pipeline", "g_only"],
                   help="pipeline: use V-band when available and GP stitch; g_only: g-band only, no V-band, no GP correction.",
    )
    p.add_argument("--extension",
                   "-e",
                   type=str,
                   default=None,
                   help="Light curve file extension (e.g., dat, dat2, dat3). Default: dat3 (from config)")
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
    file_ext: str,
    stats: dict | None = None,
) -> Iterator[tuple[str, SourceMeta]]:
    """Yield light-curve jobs lazily to avoid materializing a whole mag bin.

    If ``stats`` is provided, it will be mutated in-place with per-directory
    attrition counters so the caller can print a diagnostic summary once
    iteration is complete. Without this, several drop paths (wrong extension,
    file stem not in index, stale checkpoint) are silent and can produce
    surprisingly low yields (e.g. 80 lc_cal dirs -> 5 LCs) without any signal.
    """
    if stats is not None:
        stats.setdefault("dirs_seen", 0)
        stats.setdefault("dirs_missing_index", 0)
        stats.setdefault("dirs_zero_globbed", 0)
        stats.setdefault("dirs_zero_yielded", 0)
        stats.setdefault("globbed", 0)
        stats.setdefault("yielded", 0)
        stats.setdefault("dropped_already_processed", 0)
        stats.setdefault("dropped_non_int_stem", 0)
        stats.setdefault("dropped_no_meta_match", 0)
        stats.setdefault("ext_observed", {})
        stats.setdefault("per_dir_warnings", [])

    for lc_dir in lc_dirs:
        match = IDX_PATTERN.search(lc_dir.name)
        if match is None:
            continue
        if stats is not None:
            stats["dirs_seen"] += 1
        x = int(match.group(1))
        index_path = mag_bin_dir / f"index{x}.csv"

        if not index_path.exists():
            print(f"Skipping lc{x}_cal: missing index{x}.csv")
            if stats is not None:
                stats["dirs_missing_index"] += 1
            continue

        meta_by_id = read_index_map(index_path)

        dir_globbed = 0
        dir_yielded = 0
        dir_dropped_processed = 0
        dir_dropped_non_int = 0
        dir_dropped_no_meta = 0

        for file_path in sorted(lc_dir.glob(f"*.{file_ext}")):
            dir_globbed += 1
            file_path_str = str(file_path)
            if file_path_str in processed_files:
                dir_dropped_processed += 1
                continue
            try:
                target = int(file_path.stem)
            except ValueError:
                dir_dropped_non_int += 1
                continue
            meta = meta_by_id.get(target)
            if meta is None:
                dir_dropped_no_meta += 1
                continue
            dir_yielded += 1
            yield file_path_str, meta

        if stats is not None:
            stats["globbed"] += dir_globbed
            stats["yielded"] += dir_yielded
            stats["dropped_already_processed"] += dir_dropped_processed
            stats["dropped_non_int_stem"] += dir_dropped_non_int
            stats["dropped_no_meta_match"] += dir_dropped_no_meta
            if dir_globbed == 0:
                stats["dirs_zero_globbed"] += 1
                # Sample sibling extensions so we can tell whether the dir is
                # truly empty vs. has files under a different extension.
                try:
                    sibling_exts: dict[str, int] = {}
                    for p in lc_dir.iterdir():
                        if p.is_file():
                            ext = p.suffix.lstrip(".")
                            sibling_exts[ext] = sibling_exts.get(ext, 0) + 1
                    if sibling_exts:
                        for ext, n in sibling_exts.items():
                            stats["ext_observed"][ext] = (
                                stats["ext_observed"].get(ext, 0) + n
                            )
                        stats["per_dir_warnings"].append(
                            f"{lc_dir.name}: 0 *.{file_ext} files (other exts: "
                            + ", ".join(
                                f"{ext}={n}" for ext, n in sorted(sibling_exts.items())
                            )
                            + f"; index_rows={len(meta_by_id)})"
                        )
                    else:
                        stats["per_dir_warnings"].append(
                            f"{lc_dir.name}: directory is empty"
                            f" (index_rows={len(meta_by_id)})"
                        )
                except OSError as e:
                    stats["per_dir_warnings"].append(
                        f"{lc_dir.name}: could not list contents: {e}"
                    )
            elif dir_yielded == 0:
                stats["dirs_zero_yielded"] += 1
                stats["per_dir_warnings"].append(
                    f"{lc_dir.name}: globbed {dir_globbed} *.{file_ext} files but "
                    f"yielded 0 (no_meta={dir_dropped_no_meta}, "
                    f"non_int={dir_dropped_non_int}, "
                    f"already_processed={dir_dropped_processed}, "
                    f"index_rows={len(meta_by_id)})"
                )


def read_lc_dat2_fast(asassn_id: str, path: str, *, include_v: bool, file_ext: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    lc_path = os.path.join(path, f"{asassn_id}.{file_ext}")
    if not os.path.exists(lc_path):
        raise FileNotFoundError(f"Light curve file not found: {lc_path}")

    columns = ["JD", "mag", "error", "v_g_band", "saturated"]
    df = pd.read_csv(
        lc_path,
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

    Here ``coeff1`` and ``coeff2`` retain their legacy column names, but they now
    store the true quadratic-fit coefficients:
      coeffs = polyfit(x, y, 2) gives [a, b, c]
      quadratic_slope = a
      coeff1 = b
      coeff2 = c

    ``max_diff`` is the maximum fitted magnitude difference across the observed
    season-index range, including the quadratic vertex when it falls inside the
    sampled interval.
    """
    # Linear slope
    lin = np.polyfit(indexes, meds, 1)
    lin_slope = float(lin[0])

    # Quadratic
    quad = np.polyfit(indexes, meds, 2)
    a = float(quad[0])
    b = float(quad[1])
    c = float(quad[2])

    # Evaluate the fitted quadratic over the observed season-index span.
    x0 = float(indexes[0])
    x1 = float(indexes[-1])

    def _quad(x: float) -> float:
        return a * x * x + b * x + c

    m0 = _quad(x0)
    m1 = _quad(x1)

    # Handle near-linear case safely
    if np.isclose(a, 0.0):
        diff = abs(m1 - m0)
        return lin_slope, a, b, c, float(diff)

    vertex_x = -b / (2.0 * a)
    vertex_y = _quad(vertex_x)

    if (vertex_x > x0) and (vertex_x < x1):
        m1m0 = abs(m1 - m0)
        m1me = abs(m1 - vertex_y)
        m0me = abs(m0 - vertex_y)
        diff = max(m1m0, m1me, m0me)
    else:
        diff = abs(m1 - m0)

    return lin_slope, a, b, c, float(diff)


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


def _finite_sorted_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return np.array([], dtype=float), np.array([], dtype=float)
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    return x[order], y[order]


def _day_window_label(days: float) -> str:
    rounded = int(round(float(days)))
    if np.isclose(float(days), rounded):
        return f"{rounded}d"
    return f"{str(float(days)).replace('.', 'p')}d"


def _empty_rolling_smooth_window_features() -> dict[str, float | int]:
    return {
        "p95_p5": np.nan,
        "smooth_var": np.nan,
        "resid_var": np.nan,
        "long_short_var_ratio": np.nan,
        "n_points": 0,
    }


def _compute_rolling_smooth_window_features(
    t: np.ndarray,
    y: np.ndarray,
    *,
    window_days: float,
    min_points: int,
) -> dict[str, float | int]:
    out = _empty_rolling_smooth_window_features()
    n = y.size
    if n < int(min_points) or float(window_days) <= 0:
        return out

    half_window = float(window_days) / 2.0
    smooth = np.full(n, np.nan, dtype=float)
    left = 0
    right = 0

    for i, t_i in enumerate(t):
        while left < n and t[left] < t_i - half_window:
            left += 1
        while right < n and t[right] <= t_i + half_window:
            right += 1
        if right - left >= int(min_points):
            smooth[i] = float(np.median(y[left:right]))

    valid = np.isfinite(smooth)
    n_smooth = int(valid.sum())
    out["n_points"] = n_smooth
    if n_smooth < 2:
        return out

    smooth_valid = smooth[valid]
    residual = y[valid] - smooth_valid
    out["p95_p5"] = float(np.percentile(smooth_valid, 95) - np.percentile(smooth_valid, 5))
    out["smooth_var"] = float(np.var(smooth_valid, ddof=1))
    if residual.size >= 2:
        resid_var = float(np.var(residual, ddof=1))
        out["resid_var"] = resid_var
        if np.isfinite(resid_var) and resid_var > 0:
            out["long_short_var_ratio"] = float(out["smooth_var"] / resid_var)

    return out


def compute_rolling_smooth_features(
    JD: np.ndarray,
    mag: np.ndarray,
    *,
    window_days: float = LTV_SMOOTH_WINDOW_DAYS,
    windows_days: tuple[float, ...] = LTV_SMOOTH_WINDOWS_DAYS,
    min_points: int = LTV_SMOOTH_MIN_POINTS,
) -> dict[str, float | int]:
    """Compute raw-light-curve slow-component features across smoothing windows."""
    t, y = _finite_sorted_xy(JD, mag)
    windows: list[float] = []
    for item in tuple(windows_days) + (float(window_days),):
        value = float(item)
        if value <= 0:
            continue
        if not any(np.isclose(value, existing) for existing in windows):
            windows.append(value)

    out: dict[str, float | int] = {}
    per_window: dict[str, dict[str, float | int]] = {}
    for value in windows:
        label = _day_window_label(value)
        features = _compute_rolling_smooth_window_features(
            t,
            y,
            window_days=value,
            min_points=min_points,
        )
        per_window[label] = features
        for key, feature_value in features.items():
            out[f"smooth_{label}_{key}"] = feature_value

    alias_label = _day_window_label(float(window_days))
    alias_features = per_window.get(alias_label, _empty_rolling_smooth_window_features())
    out["smooth_p95_p5"] = alias_features["p95_p5"]
    out["smooth_var"] = alias_features["smooth_var"]
    out["resid_var"] = alias_features["resid_var"]
    out["long_short_var_ratio"] = alias_features["long_short_var_ratio"]
    out["smooth_n_points"] = alias_features["n_points"]

    return out


def compute_theil_sen_trend(t_years: np.ndarray, meds: np.ndarray) -> dict[str, float]:
    """Compute robust monotonic trend diagnostics from seasonal medians."""
    out = {
        "theil_sen_slope_mag_per_year": np.nan,
        "theil_sen_intercept_mag": np.nan,
        "theil_sen_low_slope_mag_per_year": np.nan,
        "theil_sen_high_slope_mag_per_year": np.nan,
    }
    x, y = _finite_sorted_xy(t_years, meds)
    if y.size < 2 or np.allclose(x, x[0]):
        return out

    try:
        slope, intercept, low_slope, high_slope = sp_stats.theilslopes(y, x)
    except Exception:
        return out

    out["theil_sen_slope_mag_per_year"] = float(slope)
    out["theil_sen_intercept_mag"] = float(intercept)
    out["theil_sen_low_slope_mag_per_year"] = float(low_slope)
    out["theil_sen_high_slope_mag_per_year"] = float(high_slope)
    return out


def _seasonal_measure_errors(meds: np.ndarray, meds_err: np.ndarray | None) -> np.ndarray:
    y = np.asarray(meds, dtype=float)
    err = np.full(y.shape, np.nan, dtype=float)
    if meds_err is not None:
        raw = np.asarray(meds_err, dtype=float)
        if raw.shape == y.shape:
            err = raw

    positive = err[np.isfinite(err) & (err > 0)]
    scatter = float(mad_std(y[np.isfinite(y)])) if np.isfinite(y).sum() >= 2 else np.nan
    candidates = [0.02]
    if positive.size:
        candidates.append(float(np.median(positive)))
    if np.isfinite(scatter) and scatter > 0:
        candidates.append(float(0.1 * scatter))
    fallback = max(candidates)
    return np.where(np.isfinite(err) & (err > 0), err, fallback)


def compute_bayesian_block_features(
    t_years: np.ndarray,
    meds: np.ndarray,
    meds_err: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Summarize Bayesian Blocks state changes in seasonal medians."""
    out: dict[str, float | int] = {
        "bb_n_blocks": 0,
        "bb_n_change_points": 0,
        "bb_range_mag": np.nan,
        "bb_largest_jump_mag": np.nan,
        "bb_max_block_offset_mag": np.nan,
    }
    x, y = _finite_sorted_xy(t_years, meds)
    if y.size < 2 or np.allclose(x, x[0]):
        return out

    if meds_err is None:
        yerr_sorted = None
    else:
        raw_x = np.asarray(t_years, dtype=float)
        raw_y = np.asarray(meds, dtype=float)
        raw_err = np.asarray(meds_err, dtype=float)
        mask = np.isfinite(raw_x) & np.isfinite(raw_y)
        if raw_err.shape == raw_y.shape:
            order = np.argsort(raw_x[mask])
            yerr_sorted = raw_err[mask][order]
        else:
            yerr_sorted = None
    sigma = _seasonal_measure_errors(y, yerr_sorted)

    try:
        edges = bayesian_blocks(x, y, sigma=sigma, fitness="measures")
    except Exception:
        return out

    if edges.size < 2:
        return out

    block_medians: list[float] = []
    for i in range(edges.size - 1):
        if i == edges.size - 2:
            mask = (x >= edges[i]) & (x <= edges[i + 1])
        else:
            mask = (x >= edges[i]) & (x < edges[i + 1])
        if np.any(mask):
            block_medians.append(float(np.median(y[mask])))

    if not block_medians:
        return out

    blocks = np.asarray(block_medians, dtype=float)
    out["bb_n_blocks"] = int(blocks.size)
    out["bb_n_change_points"] = int(max(0, blocks.size - 1))
    out["bb_range_mag"] = float(np.max(blocks) - np.min(blocks))
    out["bb_max_block_offset_mag"] = float(np.max(np.abs(blocks - np.median(y))))
    if blocks.size >= 2:
        out["bb_largest_jump_mag"] = float(np.max(np.abs(np.diff(blocks))))
    return out


def _local_linear_prediction(x: np.ndarray, y: np.ndarray, x0: float, weights: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    if int(valid.sum()) == 0:
        return np.nan

    xv = x[valid] - float(x0)
    yv = y[valid]
    wv = weights[valid]
    sw = float(np.sum(wv))
    if sw <= 0:
        return np.nan
    if int(valid.sum()) < 2 or np.allclose(xv, xv[0]):
        return float(np.sum(wv * yv) / sw)

    sx = float(np.sum(wv * xv))
    sy = float(np.sum(wv * yv))
    sxx = float(np.sum(wv * xv * xv))
    sxy = float(np.sum(wv * xv * yv))
    denom = sw * sxx - sx * sx
    if abs(denom) < 1e-12:
        return float(sy / sw)
    return float((sxx * sy - sx * sxy) / denom)


def _lowess_smooth(
    x: np.ndarray,
    y: np.ndarray,
    *,
    frac: float = LTV_LOWESS_FRAC,
    robust_iters: int = LTV_LOWESS_ROBUST_ITERS,
) -> np.ndarray:
    x, y = _finite_sorted_xy(x, y)
    n = y.size
    if n < 2:
        return np.full(n, np.nan, dtype=float)

    k = int(np.ceil(float(frac) * n))
    k = min(n, max(2, k))
    robust_weights = np.ones(n, dtype=float)
    fitted = np.full(n, np.nan, dtype=float)

    for iteration in range(max(0, int(robust_iters)) + 1):
        for i, x_i in enumerate(x):
            dist = np.abs(x - x_i)
            bandwidth = float(np.partition(dist, k - 1)[k - 1])
            if bandwidth <= 0:
                base_weights = (dist == 0).astype(float)
            else:
                u = dist / bandwidth
                base_weights = np.where(u < 1.0, (1.0 - u**3) ** 3, 0.0)
            weights = base_weights * robust_weights
            fitted[i] = _local_linear_prediction(x, y, x_i, weights)

        if iteration >= int(robust_iters):
            break
        resid = y - fitted
        finite_resid = resid[np.isfinite(resid)]
        if finite_resid.size == 0:
            break
        scale = 6.0 * float(np.median(np.abs(finite_resid - np.median(finite_resid))))
        if not np.isfinite(scale) or scale <= 0:
            break
        u = resid / scale
        robust_weights = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)

    return fitted


def compute_lowess_features(
    t_years: np.ndarray,
    meds: np.ndarray,
    *,
    frac: float = LTV_LOWESS_FRAC,
    robust_iters: int = LTV_LOWESS_ROBUST_ITERS,
) -> dict[str, float]:
    """Compute lightweight LOWESS diagnostics on seasonal medians."""
    out = {
        "lowess_p95_p5": np.nan,
        "lowess_resid_std": np.nan,
        "lowess_max_abs_resid": np.nan,
    }
    x, y = _finite_sorted_xy(t_years, meds)
    if y.size < 2:
        return out

    fitted = _lowess_smooth(x, y, frac=frac, robust_iters=robust_iters)
    valid = np.isfinite(fitted) & np.isfinite(y)
    if int(valid.sum()) < 2:
        return out

    fit = fitted[valid]
    resid = y[valid] - fit
    out["lowess_p95_p5"] = float(np.percentile(fit, 95) - np.percentile(fit, 5))
    out["lowess_resid_std"] = float(np.std(resid, ddof=1)) if resid.size >= 2 else np.nan
    out["lowess_max_abs_resid"] = float(np.max(np.abs(resid))) if resid.size else np.nan
    return out


def compute_variogram_features(t_years: np.ndarray, meds: np.ndarray) -> dict[str, float]:
    """Compute coarse seasonal-median variogram features."""
    out = {
        "variogram_short_mag2": np.nan,
        "variogram_mid_mag2": np.nan,
        "variogram_long_mag2": np.nan,
        "variogram_long_short_ratio": np.nan,
        "variogram_slope": np.nan,
    }
    x, y = _finite_sorted_xy(t_years, meds)
    if y.size < 3:
        return out

    dt_parts = []
    dm2_parts = []
    for i in range(y.size - 1):
        dt_parts.append(x[i + 1 :] - x[i])
        dm2_parts.append(np.square(y[i + 1 :] - y[i]))
    dt = np.concatenate(dt_parts)
    dm2 = np.concatenate(dm2_parts)
    valid = np.isfinite(dt) & np.isfinite(dm2) & (dt > 0)
    dt = dt[valid]
    dm2 = dm2[valid]
    if dt.size == 0:
        return out

    bins = {
        "short": dt < 1.5,
        "mid": (dt >= 1.5) & (dt < 3.5),
        "long": dt >= 3.5,
    }
    binned: list[tuple[float, float]] = []
    for label, mask in bins.items():
        if np.any(mask):
            value = float(np.median(dm2[mask]))
            out[f"variogram_{label}_mag2"] = value
            tau = float(np.median(dt[mask]))
            if np.isfinite(value) and value > 0 and np.isfinite(tau) and tau > 0:
                binned.append((tau, value))

    short = out["variogram_short_mag2"]
    long = out["variogram_long_mag2"]
    if np.isfinite(short) and np.isfinite(long) and float(short) > 0:
        out["variogram_long_short_ratio"] = float(long) / float(short)

    if len(binned) >= 2:
        tau = np.asarray([item[0] for item in binned], dtype=float)
        val = np.asarray([item[1] for item in binned], dtype=float)
        out["variogram_slope"] = float(np.polyfit(np.log10(tau), np.log10(val), 1)[0])

    return out


_BINNED_SF_LABELS = ("30d", "100d", "300d", "1000d", "3000d")


def _median_time_bins(
    JD: np.ndarray,
    mag: np.ndarray,
    *,
    bin_days: float,
) -> tuple[np.ndarray, np.ndarray]:
    t, y = _finite_sorted_xy(JD, mag)
    if y.size == 0 or float(bin_days) <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    bin_index = np.floor((t - float(t.min())) / float(bin_days)).astype(np.int64)
    unique_bins = np.unique(bin_index)
    bin_times = []
    bin_mags = []
    for idx in unique_bins:
        mask = bin_index == idx
        if np.any(mask):
            bin_times.append(float(np.median(t[mask])))
            bin_mags.append(float(np.median(y[mask])))
    return np.asarray(bin_times, dtype=float), np.asarray(bin_mags, dtype=float)


def compute_binned_structure_function_features(
    JD: np.ndarray,
    mag: np.ndarray,
    *,
    bin_days: float = LTV_BINNED_SF_BIN_DAYS,
    lag_bins_days: tuple[tuple[float, float], ...] = LTV_BINNED_SF_LAG_BINS_DAYS,
) -> dict[str, float | int]:
    """Compute lag-binned structure-function features from 30-day medians."""
    out: dict[str, float | int] = {"binned_sf_n_bins": 0}
    labels = _BINNED_SF_LABELS[: len(lag_bins_days)]
    for label in labels:
        out[f"binned_sf_{label}_mag2"] = np.nan
    for label in ("300d", "1000d", "3000d"):
        out[f"binned_sf_{label}_30d_ratio"] = np.nan
    out["binned_sf_slope"] = np.nan

    t, y = _median_time_bins(JD, mag, bin_days=bin_days)
    out["binned_sf_n_bins"] = int(y.size)
    if y.size < 2:
        return out

    dt_parts = []
    dm2_parts = []
    for i in range(y.size - 1):
        dt_parts.append(t[i + 1 :] - t[i])
        dm2_parts.append(np.square(y[i + 1 :] - y[i]))
    dt = np.concatenate(dt_parts)
    dm2 = np.concatenate(dm2_parts)
    valid = np.isfinite(dt) & np.isfinite(dm2) & (dt > 0)
    dt = dt[valid]
    dm2 = dm2[valid]
    if dt.size == 0:
        return out

    binned_for_slope: list[tuple[float, float]] = []
    for label, (lo, hi) in zip(labels, lag_bins_days):
        mask = (dt >= float(lo)) & (dt < float(hi))
        if not np.any(mask):
            continue
        value = float(np.median(dm2[mask]))
        out[f"binned_sf_{label}_mag2"] = value
        tau = float(np.median(dt[mask]))
        if np.isfinite(value) and value > 0 and np.isfinite(tau) and tau > 0:
            binned_for_slope.append((tau, value))

    base = out.get("binned_sf_30d_mag2", np.nan)
    if np.isfinite(base) and float(base) > 0:
        for label in ("300d", "1000d", "3000d"):
            value = out.get(f"binned_sf_{label}_mag2", np.nan)
            if np.isfinite(value):
                out[f"binned_sf_{label}_30d_ratio"] = float(value) / float(base)

    if len(binned_for_slope) >= 2:
        tau = np.asarray([item[0] for item in binned_for_slope], dtype=float)
        value = np.asarray([item[1] for item in binned_for_slope], dtype=float)
        out["binned_sf_slope"] = float(np.polyfit(np.log10(tau), np.log10(value), 1)[0])

    return out


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

    if (
        season_times.size >= 2
        and not np.allclose(season_times, season_times[0], equal_nan=True)
        and not np.allclose(meds, meds[0], equal_nan=True)
    ):
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
    finite_detrended = mag_detrended[np.isfinite(mag_detrended)]
    if finite_detrended.size < 2 or np.allclose(finite_detrended, finite_detrended[0]):
        return result
    
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
    # Remove file extension dynamically based on config
    asassn_id = os.path.splitext(basename)[0]
    target = meta.asas_sn_id
    ra_val = meta.ra_deg
    p_mag = meta.pstarrs_g_mag

    dir_path = os.path.dirname(path)
    df_g, df_v = read_lc_dat2_fast(asassn_id, dir_path, include_v=(cfg.band_mode != "g_only"), file_ext=cfg.file_ext)

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
    indexes, meds, meds_err = season_medians_with_gap_indices(
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
    season_stats = {
        f"ltv_{key}": value
        for key, value in compute_season_diagnostics(meds, t_years, season_spans, season_counts).items()
    }
    trend_stats = {
        f"ltv_{key}": value
        for key, value in compute_time_trend_diagnostics(t_years, meds).items()
    }
    smooth_stats = {
        f"ltv_{key}": value
        for key, value in compute_rolling_smooth_features(JD, mag).items()
    }
    robust_trend_stats = {
        f"ltv_{key}": value
        for key, value in compute_theil_sen_trend(t_years, meds).items()
    }
    bayesian_block_stats = {
        f"ltv_{key}": value
        for key, value in compute_bayesian_block_features(t_years, meds, meds_err).items()
    }
    lowess_stats = {
        f"ltv_{key}": value
        for key, value in compute_lowess_features(t_years, meds).items()
    }
    variogram_stats = {
        f"ltv_{key}": value
        for key, value in compute_variogram_features(t_years, meds).items()
    }
    binned_sf_stats = {
        f"ltv_{key}": value
        for key, value in compute_binned_structure_function_features(JD, mag).items()
    }

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
        "asas_sn_id": target,
        "candidate_id": f"ltv_{target}",
        "timescale": "ltv",
        "ra": ra_val,
        "dec": meta.dec_deg,
        "baseline_mag": p_mag,
        **basic_stats,
        "ltv_median": lc_median,
        "ltv_median_err": lc_mad,
        "ltv_dispersion": lc_dispersion,
        "ltv_slope": lin_slope,
        "ltv_slope_quad": quad_slope,
        "ltv_coeff1": c1,
        "ltv_coeff2": c2,
        "ltv_max_diff": diff,
        **season_stats,
        **trend_stats,
        **smooth_stats,
        **robust_trend_stats,
        **bayesian_block_stats,
        **lowess_stats,
        **variogram_stats,
        **binned_sf_stats,
        "ltv_vg_has_v": vg_has_v,
        "ltv_vg_overlap_days": vg_overlap_days,
        "ltv_vg_overlap_fraction": vg_overlap_frac,
        "ltv_vg_offset_applied": vg_offset_applied,
        "ltv_ls_period": ls_result["ls_period"],
        "ltv_ls_power": ls_result["ls_power"],
        "ltv_ls_fap": ls_result["ls_fap"],
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
        df_chunk = to_layer_first_frame(pd.DataFrame(chunk_results))
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


def _print_iter_stats_summary(stats: dict, file_ext: str, lc_dirs_total: int) -> None:
    """Print a diagnostic summary so that silent attrition in
    ``iter_light_curve_jobs`` (e.g. wrong --extension, file-stem/index ID
    mismatches, stale checkpoints) becomes visible. This is the chunk-level
    analog of the visibility we added in events.py for the STV path; here
    drops happen at the *job submission* layer rather than the *write* layer,
    but the failure mode -- a mag bin silently producing far fewer candidates
    than it should -- looks the same to the user.
    """
    if not stats:
        return

    dirs_seen = stats.get("dirs_seen", 0)
    yielded = stats.get("yielded", 0)
    globbed = stats.get("globbed", 0)
    print(
        f"[ltv-core iter] mag-bin scan: lc_dirs_total={lc_dirs_total} "
        f"dirs_seen={dirs_seen} "
        f"missing_index={stats.get('dirs_missing_index', 0)} "
        f"zero_globbed={stats.get('dirs_zero_globbed', 0)} "
        f"zero_yielded={stats.get('dirs_zero_yielded', 0)} | "
        f"globbed(*.{file_ext})={globbed} yielded={yielded} "
        f"dropped(no_meta)={stats.get('dropped_no_meta_match', 0)} "
        f"dropped(non_int)={stats.get('dropped_non_int_stem', 0)} "
        f"dropped(checkpoint)={stats.get('dropped_already_processed', 0)}"
    )

    ext_observed = stats.get("ext_observed") or {}
    if ext_observed and stats.get("dirs_zero_globbed", 0) > 0:
        ext_summary = ", ".join(
            f"{ext}={n}" for ext, n in sorted(ext_observed.items(), key=lambda kv: -kv[1])
        )
        print(
            f"[ltv-core iter] WARNING: {stats['dirs_zero_globbed']} of "
            f"{dirs_seen} lc_cal dirs had ZERO *.{file_ext} files. "
            f"Other extensions present in those dirs: {ext_summary}. "
            f"If you expected those, retry with --extension <ext>."
        )

    warnings = stats.get("per_dir_warnings") or []
    if warnings:
        sample = warnings[:8]
        for line in sample:
            print(f"[ltv-core iter]   - {line}")
        if len(warnings) > len(sample):
            print(
                f"[ltv-core iter]   ... and {len(warnings) - len(sample)} "
                "more directories with the same problem"
            )


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
    if cfg.overwrite:
        if output_path.exists():
            if output_path.is_dir():
                removed_any = False
                for child in output_path.glob("chunk_*.parquet*"):
                    child.unlink()
                    removed_any = True
                if removed_any:
                    print(f"Overwriting existing output chunks in {output_path}")
            else:
                output_path.unlink()
                print(f"Overwriting existing output file: {output_path}")
        if checkpoint_log.exists():
            checkpoint_log.unlink()
            print(f"Overwriting checkpoint log: {checkpoint_log}")

    if checkpoint_log.exists() and not cfg.overwrite:
        print(f"Resume mode: loading checkpoint from {checkpoint_log}")
        with open(checkpoint_log, "r") as f:
            processed_files = set(line.strip() for line in f)
        print(f"Found {len(processed_files)} previously processed files")

    max_in_flight = max(1, cfg.workers * 2)
    iter_stats: dict = {}
    job_iter = iter_light_curve_jobs(
        mag_bin_dir, lc_dirs, processed_files, cfg.file_ext, stats=iter_stats
    )

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
            _print_iter_stats_summary(iter_stats, cfg.file_ext, len(lc_dirs))
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
    _print_iter_stats_summary(iter_stats, cfg.file_ext, len(lc_dirs))


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
