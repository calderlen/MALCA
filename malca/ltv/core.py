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
import csv
import multiprocessing as mp
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.stats import mad_std
from astropy.table import Table
from astropy.timeseries import LombScargle
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
from malca.config.config_pipeline import MAG_BINS
from malca.config.config_io import PARQUET_OUTPUT_COMPRESSION
from malca.utils import read_lc_dat2, clean_lc

from malca.ltv.optim import (
    _detrend_fast,
    _season_medians_fast,
    _polyfit_linear_fast,
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
    # Parallel processing options
    workers: int
    chunk_size: int
    output_format: str
    resume: bool
    overwrite: bool


def _build_config(a, mag_bin: str) -> Config:
    """Build a Config for a single mag bin from parsed args."""
    root = Path(a.root)
    out = a.output
    if out is None:
        LTV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = str(LTV_OUTPUT_DIR / f"LTvar{mag_bin.replace('_','-')}.parquet")
    output = Path(out)

    resume = bool(a.resume or a.overwrite)

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
        workers=int(a.workers),
        chunk_size=int(a.chunk_size),
        output_format=str(a.output_format),
        resume=resume,
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
                   help="Combined output CSV (default: LTvar<MAG>.csv)")
    p.add_argument("--dspring",
                   type=float,
                   default=LTV_DSPRING)
    p.add_argument("--ra-is-deg",
                    action="store_true",
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
    p.add_argument("--workers",
                   type=int,
                   default=LTV_WORKERS,
                   help="Number of parallel workers (default: 10)")
    p.add_argument("--chunk-size",
                   type=int,
                   default=LTV_CORE_CHUNK_SIZE,
                   help="Number of results to accumulate before writing (default: 10000)")
    p.add_argument("--output-format",
                   type=str,
                   default="parquet",
                   choices=["csv", "parquet", "parquet_chunk"],
                   help="Output format (default: csv)")
    p.add_argument("--resume",
                   action="store_true",
                   help="Enable checkpointing to resume interrupted runs")
    p.add_argument("-o", "--overwrite",
                   action="store_true",
                   help="Overwrite existing checkpoint log when resuming (implies --resume)")

    a = p.parse_args()

    run_all = a.mag_bin == "all"

    if run_all:
        if a.output is not None:
            p.error("--output cannot be used with --all (each mag bin auto-resolves its own output path)")
        configs = [_build_config(a, mb) for mb in MAG_BINS]
    else:
        configs = [_build_config(a, a.mag_bin)]

    return configs, run_all


def read_index_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


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
    id_df: pd.DataFrame,
    cfg: Config,
) -> dict | None:
    basename = os.path.basename(path)
    asassn_id = basename.replace('.dat2', '')
    target = int(asassn_id)

    rows = id_df.loc[id_df["asas_sn_id"] == target]
    if rows.empty:
        return None
    row = rows.iloc[0]

    ra_val = float(row["ra_deg"])
    p_mag = float(row["pstarrs_g_mag"])

    dir_path = os.path.dirname(path)
    df_g, df_v = read_lc_dat2(asassn_id, dir_path)

    if df_g.empty:
        return None
    df = filter_lc_for_ltv(df_g, target)

    if df.empty:
        return None

    # V/g overlap statistics (for intercalibration failure filter)
    df_v_clean = clean_lc(df_v) if not df_v.empty else df_v
    vg_has_v, vg_overlap_days, vg_overlap_frac = compute_vg_overlap_stats(df, df_v_clean)

    JD = df["JD"].to_numpy(dtype=float)
    mag = df["mag"].to_numpy(dtype=float)
    lc_median = float(np.median(mag))
    lc_mad = float(mad_std(mag))
    lc_dispersion = float(np.ptp(mag))

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
    for s in indexes:
        sel = season_idx == s
        if not np.any(sel):
            continue
        season_times.append(np.median(JD[sel]))
        season_spans.append(JD[sel].max() - JD[sel].min())

    season_times = np.asarray(season_times, dtype=float)
    season_spans = np.asarray(season_spans, dtype=float)
    t_years = (season_times - JD.min()) / 365.25 if season_times.size > 0 else np.array([])

    lin_coeff = None
    quad_coeff = None
    if t_years.size >= 2:
        lin_coeff = tuple(np.polyfit(t_years, meds, 1))
    if t_years.size >= cfg.min_seasons_for_quadratic:
        quad_coeff = tuple(np.polyfit(t_years, meds, 2))

    avg_season_span_days = float(np.mean(season_spans)) if season_spans.size > 0 else None

    # Compute Lomb-Scargle on detrended light curve (paper: periods > 10 days)
    err = df["error"].to_numpy(dtype=float) if "error" in df.columns else None
    detrend_mode = "quadratic" if quad_coeff is not None else "linear"
    max_period_days = avg_season_span_days if avg_season_span_days is not None else LTV_LS_MAX_PERIOD_DAYS
    ls_result = compute_lomb_scargle(
        JD, mag, err,
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
        "dec_deg": float(row["dec_deg"]) if "dec_deg" in row.index else np.nan,
        "Pstarss gmag": p_mag,
        "Median": lc_median,
        "Median_err": lc_mad,
        "Dispersion": lc_dispersion,
        "Slope": lin_slope,
        "Quad Slope": quad_slope,
        "coeff1": c1,
        "coeff2": c2,
        "max diff": diff,
        "vg_has_v": vg_has_v,
        "vg_overlap_days": vg_overlap_days,
        "vg_overlap_fraction": vg_overlap_frac,
        "ls_period": ls_result["ls_period"],
        "ls_power": ls_result["ls_power"],
        "ls_fap": ls_result["ls_fap"],
        "lc_path": path,
    }


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


class CsvWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.columns = None
        if self.path.exists() and self.path.stat().st_size > 0:
            try:
                self.columns = pd.read_csv(self.path, nrows=0).columns.tolist()
            except Exception:
                self.columns = None

    def write_chunk(self, chunk_results):
        if not chunk_results:
            return
        df_chunk = pd.DataFrame(chunk_results)
        if self.columns is None:
            self.columns = list(df_chunk.columns)
        df_chunk = df_chunk.reindex(columns=self.columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = not self.path.exists() or self.path.stat().st_size == 0
        df_chunk.to_csv(self.path, mode="a", header=header, index=False)

    def close(self):
        return


class ParquetChunkWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.append = self.path.exists() and self.path.stat().st_size > 0

    def write_chunk(self, chunk_results):
        if not chunk_results:
            return
        df_chunk = pd.DataFrame(chunk_results)
        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.append:
            existing = pq.read_table(self.path)
            table = pa.concat_tables([existing, table])
        pq.write_table(table, self.path, compression=PARQUET_OUTPUT_COMPRESSION)
        self.append = True

    def close(self):
        return


class ParquetDatasetWriter:
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


def make_writer(path: Path | None, fmt: str):
    if path is None:
        return None
    if fmt == "csv":
        return CsvWriter(path)
    elif fmt == "parquet":
        return ParquetChunkWriter(path)
    elif fmt == "parquet_chunk":
        return ParquetDatasetWriter(path)
    else:
        raise ValueError(f"Unknown output format: {fmt}")


def run_mag_bin(cfg: Config) -> None:
    """Run the full LTV pipeline for a single magnitude bin."""
    mag_bin_dir = cfg.root / cfg.mag_bin
    lc_dirs = sorted(mag_bin_dir.glob("lc*_cal"))
    lc_dirs = [d for d in lc_dirs if d.is_dir() and IDX_PATTERN.search(d.name)]

    print(f"Processing mag_bin={cfg.mag_bin}: found {len(lc_dirs)} lc_cal directories")
    print(f"Workers: {cfg.workers}, Chunk size: {cfg.chunk_size}, Format: {cfg.output_format}")

    output_path = Path(cfg.output)
    checkpoint_log = output_path.with_name(f"{output_path.stem}_PROCESSED.txt") if cfg.resume else None

    processed_files = set()
    if checkpoint_log and checkpoint_log.exists() and cfg.overwrite:
        try:
            with open(checkpoint_log, "w"):
                pass
            print(f"Overwriting checkpoint log: {checkpoint_log}")
        except Exception as e:
            print(f"Warning: could not overwrite checkpoint log {checkpoint_log}: {e}")

    if checkpoint_log and checkpoint_log.exists() and not cfg.overwrite:
        print(f"Resume mode: loading checkpoint from {checkpoint_log}")
        with open(checkpoint_log, "r") as f:
            processed_files = set(line.strip() for line in f)
        print(f"Found {len(processed_files)} previously processed files")

    all_files = []
    id_map = {}

    for lc_dir in lc_dirs:
        x = int(IDX_PATTERN.search(lc_dir.name).group(1))
        index_path = mag_bin_dir / f"index{x}.csv"

        if not index_path.exists():
            print(f"Skipping lc{x}_cal: missing index{x}.csv")
            continue

        id_df = read_index_csv(index_path)
        id_map[str(lc_dir)] = id_df

        csv_files = sorted(lc_dir.glob("*.dat2"))

        for file_path in csv_files:
            if str(file_path) not in processed_files:
                all_files.append((str(file_path), str(lc_dir)))

    if not all_files:
        print("No files to process (all may be completed)")
        return

    print(f"Processing {len(all_files)} light curve files")

    writer = make_writer(output_path, cfg.output_format)
    results = []
    total_written = 0

    def write_chunk(chunk_results):
        nonlocal total_written
        if not chunk_results:
            return

        writer.write_chunk(chunk_results)

        if checkpoint_log:
            with open(checkpoint_log, "a") as f:
                for row in chunk_results:
                    f.write(row.get('_path', '') + "\n")

        total_written += len(chunk_results)
        print(f"Wrote chunk: {len(chunk_results)} rows (total: {total_written})")

    with ProcessPoolExecutor(max_workers=cfg.workers) as executor:
        futures = {}
        for file_path, lc_dir in all_files:
            id_df = id_map[lc_dir]
            future = executor.submit(process_one_lc, file_path, id_df, cfg)
            futures[future] = file_path

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing LCs", unit="lc"):
            file_path = futures[future]
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
