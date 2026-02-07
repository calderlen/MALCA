"""
Post-filters that run AFTER events.py.
Most filters depend only on the output columns from events.py; the camera
median validation also reads per-camera stats from .raw2 files via path.

Filters:
7. filter_evidence_strength - require strong Bayes factors
8. filter_run_robustness - require sufficient run count and points
9. filter_morphology - require specific morphology with good BIC
10. filter_score - require minimum dipper_score (log10 event score)

Validation filters (expensive, run on candidates only):
11. validate_periodicity - bootstrap LSP to check if source is periodic
12. validate_gaia_ruwe - flag/reject high RUWE sources from Gaia
13. validate_periodic_catalog - cross-match against known periodic catalogs

Required input columns (from events.py):
    dip_bayes_factor, jump_bayes_factor,
    dip_max_log_bf_local, jump_max_log_bf_local,
    dip_run_count, jump_run_count,
    dip_max_run_points, jump_max_run_points,
    dip_max_run_cameras, jump_max_run_cameras,
    dip_best_morph, jump_best_morph,
    dip_best_delta_bic, jump_best_delta_bic,
    dipper_score (for score filter),
    path (for logging and camera median validation)
"""

from __future__ import annotations
from pathlib import Path
from time import perf_counter
import re
import time
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier


# =============================================================================
# Catalog Query Helpers (with caching)
# =============================================================================

DEFAULT_CACHE_DIR = Path("~/.cache/malca/catalogs").expanduser()


def fetch_chen2020_ztf_periodic(
    cache_dir: Path | None = None,
    force_download: bool = False,
    show_tqdm: bool = True,
) -> pd.DataFrame:
    """
    Fetch Chen+2020 ZTF periodic variable catalog from VizieR.

    VizieR ID: J/ApJS/249/18
    Contains 781,602 periodic variables with periods and classifications.

    Parameters
    ----------
    cache_dir : Path | None
        Directory to cache downloaded catalog (default: ~/.cache/malca/catalogs)
    force_download : bool
        Re-download even if cached file exists
    show_tqdm : bool
        Show progress messages

    Returns
    -------
    pd.DataFrame
        Catalog with columns: ra, dec, period, var_type
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "chen2020_ztf_periodic.parquet"

    if cache_file.exists() and not force_download:
        if show_tqdm:
            tqdm.write(f"[fetch_chen2020] Loading cached catalog from {cache_file}")
        return pd.read_parquet(cache_file)

    if show_tqdm:
        tqdm.write("[fetch_chen2020] Querying VizieR J/ApJS/249/18 (this may take a few minutes)...")

    try:
        v = Vizier(columns=["RAJ2000", "DEJ2000", "Per", "Type", "GaiaEDR3"], row_limit=-1)
        tables = v.get_catalogs("J/ApJS/249/18")

        if not tables:
            raise ValueError("No tables returned from VizieR query")

        cat = tables[0].to_pandas()

        df = pd.DataFrame({
            "ra": cat["RAJ2000"].astype(float),
            "dec": cat["DEJ2000"].astype(float),
            "period": cat["Per"].astype(float),
            "var_type": cat["Type"].astype(str),
        })

        if "GaiaEDR3" in cat.columns:
            df["gaia_id"] = pd.to_numeric(cat["GaiaEDR3"], errors="coerce").astype("Int64")
        elif "GaiaDR3" in cat.columns:
            df["gaia_id"] = pd.to_numeric(cat["GaiaDR3"], errors="coerce").astype("Int64")
        elif "GaiaDR2" in cat.columns:
            df["gaia_id"] = pd.to_numeric(cat["GaiaDR2"], errors="coerce").astype("Int64")

        df.to_parquet(cache_file, index=False)
        if show_tqdm:
            tqdm.write(f"[fetch_chen2020] Cached {len(df)} sources to {cache_file}")

        return df

    except Exception as e:
        raise RuntimeError(f"Failed to fetch Chen+2020 catalog from VizieR: {e}")


def fetch_gaia_dr3_variables(
    cache_dir: Path | None = None,
    force_download: bool = False,
    show_tqdm: bool = True,
    row_limit: int = -1,
) -> pd.DataFrame:
    """
    Fetch Gaia DR3 classified variable sources via TAP.

    Queries gaiadr3.vari_classifier_result for sources with classifications.

    Parameters
    ----------
    cache_dir : Path | None
        Directory to cache downloaded catalog (default: ~/.cache/malca/catalogs)
    force_download : bool
        Re-download even if cached file exists
    show_tqdm : bool
        Show progress messages
    row_limit : int
        Max rows to fetch (-1 for all, ~10.5M sources)

    Returns
    -------
    pd.DataFrame
        Catalog with columns: source_id, ra, dec, best_class_name
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "gaia_dr3_variables.parquet"

    if cache_file.exists() and not force_download:
        if show_tqdm:
            tqdm.write(f"[fetch_gaia_dr3_variables] Loading cached catalog from {cache_file}")
        return pd.read_parquet(cache_file)

    if show_tqdm:
        tqdm.write("[fetch_gaia_dr3_variables] Querying Gaia DR3 via TAP (this may take several minutes)...")

    try:
        limit_clause = f"TOP {row_limit}" if row_limit > 0 else ""
        query = f"""
        SELECT {limit_clause}
            v.source_id, g.ra, g.dec, v.best_class_name
        FROM gaiadr3.vari_classifier_result AS v
        JOIN gaiadr3.gaia_source AS g ON v.source_id = g.source_id
        """

        job = Gaia.launch_job_async(query)
        result = job.get_results()
        df = result.to_pandas()

        df = df.rename(columns={
            "SOURCE_ID": "source_id",
            "RA": "ra",
            "DEC": "dec",
            "BEST_CLASS_NAME": "best_class_name",
        })

        df.to_parquet(cache_file, index=False)
        if show_tqdm:
            tqdm.write(f"[fetch_gaia_dr3_variables] Cached {len(df)} sources to {cache_file}")

        return df

    except Exception as e:
        raise RuntimeError(f"Failed to fetch Gaia DR3 variables: {e}")


def fetch_gaia_dr3_ruwe(
    source_ids: list[int] | None = None,
    cache_dir: Path | None = None,
    show_tqdm: bool = True,
    batch_size: int = 10000,
    throttle_seconds: float = 1.0,
) -> pd.DataFrame:
    """
    Fetch Gaia DR3 RUWE values for specific sources.

    Results are cached locally to avoid repeated queries.

    Parameters
    ----------
    source_ids : list[int] | None
        Gaia source IDs to query
    cache_dir : Path | None
        Cache directory (default: ~/.cache/malca/catalogs)
    show_tqdm : bool
        Show progress messages
    batch_size : int
        Number of source IDs per query batch (default: 10000)
    throttle_seconds : float
        Delay between batches to avoid rate limiting (default: 1.0)

    Returns
    -------
    pd.DataFrame
        Catalog with columns: source_id, ra, dec, ruwe
    """
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "gaia_dr3_ruwe_cache.parquet"

    if source_ids is None or len(source_ids) == 0:
        raise ValueError("Must provide source_ids")

    # Load existing cache
    cached_df = pd.DataFrame()
    if cache_file.exists():
        try:
            cached_df = pd.read_parquet(cache_file)
            if show_tqdm:
                tqdm.write(f"[fetch_gaia_dr3_ruwe] Loaded {len(cached_df)} cached RUWE values")
        except Exception as e:
            if show_tqdm:
                tqdm.write(f"[fetch_gaia_dr3_ruwe] Warning: Could not load cache: {e}")

    # Determine which IDs need to be queried
    requested_ids = set(int(sid) for sid in source_ids)
    if not cached_df.empty and "source_id" in cached_df.columns:
        cached_ids = set(cached_df["source_id"].astype(int))
        ids_to_query = list(requested_ids - cached_ids)
    else:
        ids_to_query = list(requested_ids)

    if show_tqdm:
        tqdm.write(f"[fetch_gaia_dr3_ruwe] {len(requested_ids) - len(ids_to_query)} already cached, {len(ids_to_query)} to query")

    # Query missing IDs
    if ids_to_query:
        try:
            new_results = []
            n_batches = (len(ids_to_query) + batch_size - 1) // batch_size

            for i in range(0, len(ids_to_query), batch_size):
                batch = ids_to_query[i:i + batch_size]
                id_list = ",".join(str(sid) for sid in batch)
                query = f"""
                SELECT source_id, ra, dec, ruwe
                FROM gaiadr3.gaia_source
                WHERE source_id IN ({id_list})
                """
                job = Gaia.launch_job(query)
                result = job.get_results().to_pandas()
                new_results.append(result)

                batch_num = i // batch_size + 1
                if show_tqdm:
                    tqdm.write(f"[fetch_gaia_dr3_ruwe] Fetched batch {batch_num}/{n_batches} ({len(result)} results)")

                # Throttle between batches (except last)
                if batch_num < n_batches and throttle_seconds > 0:
                    time.sleep(throttle_seconds)

            # Combine new results
            if new_results:
                new_df = pd.concat(new_results, ignore_index=True)
                new_df.columns = new_df.columns.str.lower()

                # Update cache
                if not cached_df.empty:
                    cached_df = pd.concat([cached_df, new_df], ignore_index=True)
                    cached_df = cached_df.drop_duplicates(subset=["source_id"], keep="last")
                else:
                    cached_df = new_df

                # Save updated cache
                cached_df.to_parquet(cache_file, index=False)
                if show_tqdm:
                    tqdm.write(f"[fetch_gaia_dr3_ruwe] Updated cache with {len(new_df)} new entries (total: {len(cached_df)})")

        except Exception as e:
            raise RuntimeError(f"Failed querying Gaia RUWE batches: {e}")

    # Return only requested IDs
    if cached_df.empty:
        return pd.DataFrame(columns=["source_id", "ra", "dec", "ruwe"])

    result_df = cached_df[cached_df["source_id"].astype(int).isin(requested_ids)].copy()
    return result_df.reset_index(drop=True)


def log_rejections(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    filter_name: str,
    log_csv: str | Path | None,
) -> None:
    """
    Log rejected candidates to a CSV file.
    Assumes 'path' column exists (from events.py).
    """
    if log_csv is None:
        return

    before_ids = set(df_before["path"].astype(str))
    after_ids = set(df_after["path"].astype(str))
    rejected = sorted(before_ids - after_ids)
    if not rejected:
        return

    log_path = Path(log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df_log = pd.DataFrame({"path": rejected, "filter": filter_name})
    df_log.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)


def filter_evidence_strength(
    df: pd.DataFrame,
    *,
    min_bayes_factor: float = 10.0,
    require_finite_local_bf: bool = True,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Require dip_bayes_factor or jump_bayes_factor > threshold.
    Optionally require dip_max_log_bf_local or jump_max_log_bf_local to be finite.
    """
    n0 = len(df)
    pbar = tqdm(total=2, desc="filter_evidence_strength", leave=False) if show_tqdm else None

    # At least one of dip or jump BF must exceed threshold
    mask = (df["dip_bayes_factor"].fillna(0) > min_bayes_factor) | \
           (df["jump_bayes_factor"].fillna(0) > min_bayes_factor)

    # Require finite local BF if requested
    if require_finite_local_bf:
        is_finite_dip = df["dip_max_log_bf_local"].notna() & np.isfinite(df["dip_max_log_bf_local"])
        is_finite_jump = df["jump_max_log_bf_local"].notna() & np.isfinite(df["jump_max_log_bf_local"])
        mask &= (is_finite_dip | is_finite_jump)

    out = df.loc[mask].reset_index(drop=True)

    if pbar:
        pbar.update(1)

    if show_tqdm and verbose:
        tqdm.write(f"[filter_evidence_strength] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_evidence_strength", rejected_log_csv)

    if pbar:
        pbar.update(1)
        pbar.close()

    return out



# =============================================================================
# Filter 9: Run robustness
# =============================================================================

def filter_run_robustness(
    df: pd.DataFrame,
    *,
    min_run_count: int = 1,
    min_run_points: int = 2,
    min_run_cameras: int = 2,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Require dip_run_count or jump_run_count >= min_run_count.
    Require dip_max_run_points or jump_max_run_points >= min_run_points.
    Require dip_max_run_cameras or jump_max_run_cameras >= min_run_cameras.
    """
    n0 = len(df)
    pbar = tqdm(total=2, desc="filter_run_robustness", leave=False) if show_tqdm else None

    # Check run counts
    dip_count_ok = df["dip_run_count"].fillna(0) >= min_run_count
    jump_count_ok = df["jump_run_count"].fillna(0) >= min_run_count

    # Check run points
    dip_points_ok = df["dip_max_run_points"].fillna(0) >= min_run_points
    jump_points_ok = df["jump_max_run_points"].fillna(0) >= min_run_points

    # Check run cameras
    dip_cams_ok = df["dip_max_run_cameras"].fillna(0) >= min_run_cameras
    jump_cams_ok = df["jump_max_run_cameras"].fillna(0) >= min_run_cameras

    dip_ok = dip_count_ok & dip_points_ok & dip_cams_ok
    jump_ok = jump_count_ok & jump_points_ok & jump_cams_ok

    mask = dip_ok | jump_ok
    out = df.loc[mask].reset_index(drop=True)

    if pbar:
        pbar.update(1)

    if show_tqdm and verbose:
        tqdm.write(f"[filter_run_robustness] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_run_robustness", rejected_log_csv)

    if pbar:
        pbar.update(1)
        pbar.close()

    return out


# =============================================================================
# Filter 10: Morphology
# =============================================================================

def filter_morphology(
    df: pd.DataFrame,
    *,
    dip_morphology: str = "gaussian",
    jump_morphology: str = "paczynski",
    min_delta_bic: float = 10.0,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Keep runs whose best morphology is 'gaussian' for dips or 'paczynski' for jumps,
    with dip_best_delta_bic/jump_best_delta_bic >= threshold to reject noise-like runs.
    """
    n0 = len(df)
    pbar = tqdm(total=2, desc="filter_morphology", leave=False) if show_tqdm else None

    # Check morphology for dips
    dip_morph_ok = (df["dip_best_morph"].fillna("").str.lower() == dip_morphology.lower()) & \
                   (df["dip_best_delta_bic"].fillna(0) >= min_delta_bic)

    # Check morphology for jumps
    jump_morph_ok = (df["jump_best_morph"].fillna("").str.lower() == jump_morphology.lower()) & \
                    (df["jump_best_delta_bic"].fillna(0) >= min_delta_bic)

    mask = dip_morph_ok | jump_morph_ok
    out = df.loc[mask].reset_index(drop=True)

    if pbar:
        pbar.update(1)

    if show_tqdm and verbose:
        tqdm.write(f"[filter_morphology] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_morphology", rejected_log_csv)

    if pbar:
        pbar.update(1)
        pbar.close()

    return out


# =============================================================================
# Filter 10: Event score
# =============================================================================

def filter_score(
    df: pd.DataFrame,
    *,
    min_score: float = -3.0,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Require dipper_score >= min_score.

    The dipper_score is the log10 event score computed during detection
    (higher = more significant events). Candidates with -inf score
    (no valid events detected) are always rejected.

    Parameters
    ----------
    min_score : float
        Minimum log10 event score to keep (default: -3.0).
    """
    n0 = len(df)

    if "dipper_score" not in df.columns:
        if verbose:
            tqdm.write("[filter_score] WARNING: 'dipper_score' column missing, skipping filter")
        return df.copy()

    mask = df["dipper_score"].fillna(-np.inf) >= min_score
    out = df.loc[mask].reset_index(drop=True)

    if show_tqdm and verbose:
        tqdm.write(f"[filter_score] kept {len(out)}/{n0}")
    log_rejections(df, out, "filter_score", rejected_log_csv)

    return out


# =============================================================================
# Validation filters (expensive checks, run after event detection)
# =============================================================================

RAW_STATS_COLUMNS = [
    "camera",
    "median",
    "sig1_low",
    "sig1_high",
    "p90_low",
    "p90_high",
]


def _parse_mag_bin_range(mag_bin: str | None) -> tuple[float, float] | None:
    if not mag_bin:
        return None
    token = mag_bin.strip().replace("-", "_")
    parts = token.split("_")
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def _mag_bin_range_from_path(path: Path) -> tuple[float, float] | None:
    match = re.search(r"(\d+(?:\.\d+)?_\d+(?:\.\d+)?)", str(path))
    if not match:
        return None
    return _parse_mag_bin_range(match.group(1))


def _find_raw_stats_path(path: Path) -> Path:
    path = Path(path)
    if path.suffix.lower() == ".raw2":
        return path
    return path.with_suffix(".raw2")


def _read_raw_camera_stats(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        names=RAW_STATS_COLUMNS,
        comment="#",
        header=None,
    )
    for col in ("median", "sig1_low", "sig1_high", "p90_low", "p90_high"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["median"].notna()].reset_index(drop=True)
    return df



def _lsp_worker(args: tuple) -> dict:
    """
    Worker function for parallel LSP computation.
    
    Args:
        args: Tuple of (path_str, n_bootstrap, exclude_alias_periods)
        
    Returns:
        Dict with path and LSP results
    """
    from pathlib import Path as WorkerPath
    from malca.utils import read_lc_dat2
    from malca.stats import bootstrap_lomb_scargle
    
    path_str, n_bootstrap, exclude_alias_periods = args
    
    try:
        path = WorkerPath(path_str)
        asassn_id = path.stem
        dir_path = str(path.parent)
        
        dfg, dfv = read_lc_dat2(asassn_id, dir_path)
        df_lc = pd.concat([dfg, dfv], ignore_index=True)
        
        jd = df_lc["JD"].values
        mag = df_lc["mag"].values
        err = df_lc["error"].values
        
        ls_result = bootstrap_lomb_scargle(
            jd, mag, err,
            n_bootstrap=n_bootstrap,
            exclude_alias_periods=exclude_alias_periods,
        )
        
        return {
            "path": path_str,
            "lsp_power": ls_result["ls_power"],
            "lsp_period": ls_result["ls_period_days"],
            "lsp_bootstrap_sig": ls_result["ls_bootstrap_sig"],
            "lsp_is_alias": ls_result["ls_is_alias"],
            "lsp_is_significant": ls_result["ls_is_significant"],
            "error": None,
        }
    except Exception as e:
        return {
            "path": path_str,
            "lsp_power": np.nan,
            "lsp_period": np.nan,
            "lsp_bootstrap_sig": np.nan,
            "lsp_is_alias": False,
            "lsp_is_significant": False,
            "error": str(e),
        }


def validate_periodicity(
    df: pd.DataFrame,
    *,
    n_bootstrap: int = 1000,
    significance_level: float = 0.01,
    exclude_alias_periods: bool = True,
    flag_only: bool = True,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
    workers: int = 1,
    checkpoint_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Detailed periodicity validation on candidates (like ZTF paper Section 4.5).

    Uses bootstrap Lomb-Scargle periodogram with significance testing to identify:
    - Eclipsing binaries (short periods ~1 day)
    - Rotating variables (periods ~30 days)
    - Other periodic contamination

    Much more expensive than pre-filter LSP - uses bootstrap for significance.
    Only run on detected candidates, not all sources.

    Parameters
    ----------
    df : pd.DataFrame
        Candidates from events.py (must have 'path' column)
    n_bootstrap : int
        Number of bootstrap iterations for significance (default 1000)
    significance_level : float
        Significance threshold (default 0.01 = 1% like paper)
    exclude_alias_periods : bool
        Exclude known alias periods (sidereal day, lunar month, ZTF cadence)
    show_tqdm : bool
        Show progress
    rejected_log_csv : str | Path | None
        Log file for rejected candidates
    workers : int
        Number of parallel workers (default 1 = sequential)
    checkpoint_dir : str | Path | None
        Directory for checkpoint files (enables resume on restart)

    Returns
    -------
    pd.DataFrame
        Candidates without strong periodic signals

    Notes
    -----
    This implements the paper's approach:
    - Bootstrap LSP for significance levels
    - Phase-fold at best period
    - Exclude alias frequencies
    - Flag coherent periodic structure
    """
    from multiprocessing import Pool, cpu_count
    import json

    n0 = len(df)
    paths = df["path"].tolist()
    
    # Checkpoint handling
    checkpoint_file = None
    completed_results = {}
    
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_path / "lsp_checkpoint.parquet"
        
        # Load existing checkpoint if present
        if checkpoint_file.exists():
            try:
                checkpoint_df = pd.read_parquet(checkpoint_file)
                completed_results = {
                    row["path"]: row.to_dict() 
                    for _, row in checkpoint_df.iterrows()
                }
                if show_tqdm:
                    tqdm.write(f"[validate_periodicity] Loaded {len(completed_results)} cached results from checkpoint")
            except Exception as e:
                if show_tqdm:
                    tqdm.write(f"[validate_periodicity] Warning: Could not load checkpoint: {e}")
    
    # Filter to paths not already processed
    paths_to_process = [p for p in paths if p not in completed_results]
    
    if show_tqdm:
        tqdm.write(f"[validate_periodicity] {len(paths) - len(paths_to_process)}/{len(paths)} already cached, processing {len(paths_to_process)}")
    
    # Prepare worker arguments
    worker_args = [(p, n_bootstrap, exclude_alias_periods) for p in paths_to_process]
    
    # Process with multiprocessing or sequential based on workers
    new_results = []
    n_errors = 0
    
    if workers > 1 and len(paths_to_process) > 0:
        # Parallel execution
        actual_workers = min(workers, cpu_count(), len(paths_to_process))
        
        with Pool(processes=actual_workers) as pool:
            iterator = pool.imap_unordered(_lsp_worker, worker_args)
            if show_tqdm:
                iterator = tqdm(iterator, total=len(paths_to_process), desc="LSP validation")
            
            checkpoint_batch = []
            checkpoint_interval = max(100, len(paths_to_process) // 20)  # Save every 5%
            
            for result in iterator:
                new_results.append(result)
                if result["error"] is not None:
                    n_errors += 1
                
                # Batch checkpoint saves
                if checkpoint_file is not None:
                    checkpoint_batch.append(result)
                    if len(checkpoint_batch) >= checkpoint_interval:
                        _save_checkpoint(checkpoint_file, completed_results, new_results)
                        checkpoint_batch = []
    else:
        # Sequential execution (workers=1 or no paths to process)
        iterator = worker_args
        if show_tqdm and len(paths_to_process) > 0:
            iterator = tqdm(worker_args, desc="LSP validation")
        
        for args in iterator:
            result = _lsp_worker(args)
            new_results.append(result)
            if result["error"] is not None:
                n_errors += 1
    
    # Final checkpoint save
    if checkpoint_file is not None and new_results:
        _save_checkpoint(checkpoint_file, completed_results, new_results)
        if show_tqdm:
            tqdm.write(f"[validate_periodicity] Saved checkpoint with {len(completed_results) + len(new_results)} entries")
    
    # Combine cached + new results
    all_results = {**completed_results}
    for r in new_results:
        all_results[r["path"]] = r
    
    # Build output columns
    powers = []
    periods = []
    bootstrap_significances = []
    is_alias = []
    is_significant = []
    periodicity_scores = []
    keep_flags = []
    
    for path_str in paths:
        result = all_results.get(path_str, {})
        powers.append(result.get("lsp_power", np.nan))
        periods.append(result.get("lsp_period", np.nan))
        sig = result.get("lsp_bootstrap_sig", np.nan)
        bootstrap_significances.append(sig)
        alias_flag = result.get("lsp_is_alias", False)
        is_alias.append(alias_flag)
        is_significant.append(bool(result.get("lsp_is_significant", False)))

        if np.isfinite(sig):
            min_p = max(1.0 / float(max(n_bootstrap, 1)), 1e-12)
            periodicity_scores.append(float(-np.log10(np.clip(sig, min_p, 1.0))))
        else:
            periodicity_scores.append(np.nan)
        
        # Keep if NOT significantly periodic (or is an alias, or has error)
        keep = True
        if not result.get("lsp_is_alias", False) and result.get("error") is None:
            sig = result.get("lsp_bootstrap_sig", np.nan)
            if not np.isnan(sig) and sig < significance_level:
                keep = False
        keep_flags.append(keep)

    df_out = df.copy()
    df_out["lsp_power"] = powers
    df_out["lsp_period"] = periods
    df_out["lsp_bootstrap_sig"] = bootstrap_significances
    df_out["lsp_is_alias"] = is_alias
    df_out["lsp_is_significant"] = is_significant
    df_out["periodicity_score"] = periodicity_scores

    periodic_flags = [not x for x in keep_flags]
    df_out["periodic_flag"] = periodic_flags

    if flag_only:
        df_filtered = df_out.reset_index(drop=True)
    else:
        df_filtered = df_out[keep_flags].reset_index(drop=True)

    if show_tqdm:
        n_flagged = int(np.sum(periodic_flags))
        if flag_only:
            tqdm.write(f"[validate_periodicity] flagged {n_flagged}/{n0} as periodic")
        else:
            tqdm.write(f"[validate_periodicity] kept {len(df_filtered)}/{n0}")
        if n_errors > 0:
            tqdm.write(f"[validate_periodicity] {n_errors} sources had errors (kept as-is)")

    if not flag_only:
        log_rejections(df_out, df_filtered, "validate_periodicity", rejected_log_csv)

    return df_filtered


def _save_checkpoint(checkpoint_file: Path, completed: dict, new_results: list) -> None:
    """Save checkpoint to parquet file."""
    all_data = list(completed.values()) + new_results
    # Remove error column for storage
    clean_data = []
    for r in all_data:
        clean_data.append({
            "path": r["path"],
            "lsp_power": r.get("lsp_power", np.nan),
            "lsp_period": r.get("lsp_period", np.nan),
            "lsp_bootstrap_sig": r.get("lsp_bootstrap_sig", np.nan),
            "lsp_is_alias": r.get("lsp_is_alias", False),
            "lsp_is_significant": r.get("lsp_is_significant", False),
        })
    pd.DataFrame(clean_data).to_parquet(checkpoint_file, index=False)


def validate_gaia_ruwe(
    df: pd.DataFrame,
    *,
    max_ruwe: float = 1.4,
    flag_only: bool = True,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Validate candidates using Gaia RUWE (Renormalized Unit Weight Error).

    Queries Gaia DR3 via TAP for candidate coordinates.
    RUWE > 1.4 indicates potential companion (binary contamination).
    Paper identifies 5/81 candidates with high RUWE.

    Parameters
    ----------
    df : pd.DataFrame
        Candidates (must have gaia_id column)
    max_ruwe : float
        RUWE threshold (default 1.4, from paper)
    flag_only : bool
        If True, add 'ruwe' and 'high_ruwe_flag' columns but don't reject
        If False, reject sources with RUWE > max_ruwe
    show_tqdm : bool
        Show progress
    rejected_log_csv : str | Path | None
        Log file for rejected candidates

    Returns
    -------
    pd.DataFrame
        Candidates with RUWE information added

    Notes
    -----
    Paper approach:
    - RUWE ~ 1 consistent with single stars
    - RUWE > 1.4 indicates binarity
    - 5/81 candidates flagged (potential companions)
    - Still need follow-up (imaging, RV) to confirm
    """
    n0 = len(df)

    if "gaia_id" not in df.columns:
        raise ValueError("[validate_gaia_ruwe] Missing gaia_id column")

    # Get unique Gaia IDs (excluding NaN/invalid)
    valid_mask = df["gaia_id"].notna() & (df["gaia_id"] != 0)
    unique_ids = df.loc[valid_mask, "gaia_id"].astype(int).unique().tolist()

    if not unique_ids:
        if show_tqdm:
            tqdm.write("[validate_gaia_ruwe] No valid Gaia IDs - returning unchanged")
        df_out = df.copy()
        df_out["ruwe"] = np.nan
        df_out["high_ruwe_flag"] = False
        return df_out

    # Fetch RUWE from Gaia TAP by source_id
    if show_tqdm:
        tqdm.write(f"[validate_gaia_ruwe] Querying Gaia TAP for {len(unique_ids)} unique sources...")
    try:
        gaia_df = fetch_gaia_dr3_ruwe(
            source_ids=unique_ids,
            show_tqdm=show_tqdm,
        )
        if gaia_df.empty:
            if show_tqdm:
                tqdm.write("[validate_gaia_ruwe] No Gaia matches found - returning unchanged")
            df_out = df.copy()
            df_out["ruwe"] = np.nan
            df_out["high_ruwe_flag"] = False
            return df_out
    except Exception as e:
        if show_tqdm:
            tqdm.write(f"[validate_gaia_ruwe] Gaia TAP query failed: {e} - returning unchanged")
        df_out = df.copy()
        df_out["ruwe"] = np.nan
        df_out["high_ruwe_flag"] = False
        return df_out

    # Create lookup dict from Gaia results
    ruwe_lookup = dict(zip(gaia_df["source_id"].astype(int), gaia_df["ruwe"]))

    # Map RUWE values to candidates
    ruwes = []
    high_ruwe_flags = []
    for gaia_id in df["gaia_id"]:
        if pd.notna(gaia_id) and int(gaia_id) in ruwe_lookup:
            ruwe_val = float(ruwe_lookup[int(gaia_id)])
            ruwes.append(ruwe_val)
            high_ruwe_flags.append(ruwe_val > max_ruwe)
        else:
            ruwes.append(np.nan)
            high_ruwe_flags.append(False)

    df_out = df.copy()
    df_out["ruwe"] = ruwes
    df_out["high_ruwe_flag"] = high_ruwe_flags

    if flag_only:
        df_filtered = df_out
    else:
        df_filtered = df_out[~df_out["high_ruwe_flag"]].reset_index(drop=True)

    if show_tqdm:
        n_flagged = sum(high_ruwe_flags)
        tqdm.write(f"[validate_gaia_ruwe] flagged {n_flagged}/{n0} with RUWE > {max_ruwe}")
        tqdm.write(f"[validate_gaia_ruwe] kept {len(df_filtered)}/{n0}")

    if not flag_only:
        log_rejections(df_out, df_filtered, "validate_gaia_ruwe", rejected_log_csv)

    return df_filtered


def validate_periodic_catalog(
    df: pd.DataFrame,
    *,
    max_sep_arcsec: float = 3.0,
    flag_only: bool = True,
    show_tqdm: bool = False,
    verbose: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Crossmatch candidates to Chen+2020 ZTF periodic variable catalog.

    Fetches catalog from VizieR (cached locally after first download).
    Paper crossmatched to Chen+2020 and found 14 matches, including
    1 compelling BY Dra variable.

    Parameters
    ----------
    df : pd.DataFrame
        Candidates (must have gaia_id column)
    max_sep_arcsec : float
        Maximum separation for coordinate fallback (default 3 arcsec)
    flag_only : bool
        If True, add 'periodic_catalog_match' flag but don't reject
        If False, reject catalog matches
    show_tqdm : bool
        Show progress
    rejected_log_csv : str | Path | None
        Log file for rejected candidates

    Returns
    -------
    pd.DataFrame
        Candidates with periodic catalog match flags

    Notes
    -----
    Paper found:
    - 14/81 candidates matched Chen+2020 ZTF periodic catalog
    - Visual inspection showed most lacked coherent phase-folded structure
    - 1 compelling match: BY Dra variable with 2.57d period

    Suggests many catalog "periodic" sources aren't strongly periodic
    at the level detected by events.py Bayesian fitting.
    """
    n0 = len(df)

    if "gaia_id" not in df.columns:
        raise ValueError("[validate_periodic_catalog] Missing gaia_id column")

    # Fetch Chen+2020 ZTF periodic catalog from VizieR (cached)
    if show_tqdm:
        tqdm.write(f"[validate_periodic_catalog] Fetching Chen+2020 from VizieR...")
    try:
        catalog_df = fetch_chen2020_ztf_periodic(show_tqdm=show_tqdm)
        # Rename columns to match expected format
        catalog_df = catalog_df.rename(columns={"var_type": "class"})
    except Exception as e:
        if show_tqdm:
            tqdm.write(f"[validate_periodic_catalog] VizieR query failed: {e} - returning unchanged")
        df_out = df.copy()
        df_out["catalog_match"] = False
        df_out["catalog_period"] = np.nan
        df_out["catalog_class"] = ""
        return df_out

    # Check if catalog has Gaia IDs for direct matching
    use_gaia_match = "gaia_id" in catalog_df.columns and catalog_df["gaia_id"].notna().any()

    if use_gaia_match:
        if show_tqdm:
            tqdm.write(f"[validate_periodic_catalog] Matching by Gaia ID...")
        # Create lookup dict from catalog
        catalog_lookup = {}
        for _, row in catalog_df.iterrows():
            gid = row.get("gaia_id")
            if pd.notna(gid):
                catalog_lookup[int(gid)] = {
                    "period": row.get("period", np.nan),
                    "class": str(row.get("class", "")),
                }

        matches = []
        periods = []
        classes = []

        for gaia_id in df["gaia_id"]:
            if pd.notna(gaia_id) and int(gaia_id) in catalog_lookup:
                matches.append(True)
                periods.append(float(catalog_lookup[int(gaia_id)]["period"]))
                classes.append(catalog_lookup[int(gaia_id)]["class"])
            else:
                matches.append(False)
                periods.append(np.nan)
                classes.append("")
    else:
        # Fallback to coordinate crossmatch
        if show_tqdm:
            tqdm.write(f"[validate_periodic_catalog] Catalog has no Gaia IDs, falling back to coordinate match...")

        if "ra_deg" not in df.columns or "dec_deg" not in df.columns:
            raise ValueError("[validate_periodic_catalog] Need ra_deg/dec_deg for coordinate fallback")

        candidates_coords = SkyCoord(
            ra=df["ra_deg"].values * u.deg,
            dec=df["dec_deg"].values * u.deg
        )

        catalog_coords = SkyCoord(
            ra=catalog_df["ra"].values * u.deg,
            dec=catalog_df["dec"].values * u.deg
        )

        idx_catalog, sep2d, _ = candidates_coords.match_to_catalog_sky(catalog_coords)

        matches = []
        periods = []
        classes = []

        for i, (cat_idx, sep) in enumerate(zip(idx_catalog, sep2d)):
            sep_arcsec = sep.to(u.arcsec).value
            if sep_arcsec < max_sep_arcsec:
                matches.append(True)
                periods.append(float(catalog_df.iloc[cat_idx].get("period", np.nan)))
                classes.append(str(catalog_df.iloc[cat_idx].get("class", "")))
            else:
                matches.append(False)
                periods.append(np.nan)
                classes.append("")

    df_out = df.copy()
    df_out["catalog_match"] = matches
    df_out["catalog_period"] = periods
    df_out["catalog_class"] = classes

    if flag_only:
        df_filtered = df_out
    else:
        df_filtered = df_out[~df_out["catalog_match"]].reset_index(drop=True)

    if show_tqdm:
        n_matched = sum(matches)
        tqdm.write(f"[validate_periodic_catalog] matched {n_matched}/{n0} to catalog")
        tqdm.write(f"[validate_periodic_catalog] kept {len(df_filtered)}/{n0}")

    if not flag_only:
        log_rejections(df_out, df_filtered, "validate_periodic_catalog", rejected_log_csv)

    return df_filtered


# =============================================================================
# Main orchestration
# =============================================================================

def apply_post_filters(
    df: pd.DataFrame,
    *,
    # Filter 7: evidence strength
    apply_evidence_strength: bool = True,
    min_bayes_factor: float = 10.0,
    require_finite_local_bf: bool = True,
    # Filter 9: run robustness
    apply_run_robustness: bool = True,
    min_run_count: int = 1,
    min_run_points: int = 2,
    min_run_cameras: int = 2,
    # Filter 10: morphology
    apply_morphology: bool = False,
    dip_morphology: str = "gaussian",
    jump_morphology: str = "paczynski",
    min_delta_bic: float = 10.0,
    # Filter 11: event score
    apply_score: bool = True,
    min_score: float = 0.0,
    # Validation: periodicity
    apply_periodicity_validation: bool = False,
    periodicity_n_bootstrap: int = 1000,
    periodicity_significance: float = 0.01,
    periodicity_exclude_aliases: bool = True,
    periodicity_flag_only: bool = True,
    periodicity_workers: int = 1,
    periodicity_checkpoint_dir: Path | None = None,
    # Validation: Gaia RUWE
    apply_gaia_ruwe_validation: bool = True,
    gaia_max_ruwe: float = 1.4,
    gaia_flag_only: bool = True,
    # Validation: periodic catalog
    apply_periodic_catalog_validation: bool = True,
    periodic_catalog_max_sep: float = 3.0,
    periodic_catalog_flag_only: bool = True,
    # General
    show_tqdm: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Apply post-filters after running events.py.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe from events.py
    apply_* : bool
        Whether to apply each filter
    apply_periodicity_validation : bool
        Apply bootstrap LSP validation (expensive, off by default)
    apply_gaia_ruwe_validation : bool
        Apply Gaia RUWE validation (queries Gaia TAP)
    apply_periodic_catalog_validation : bool
        Apply periodic catalog crossmatch (fetches Chen+2020 from VizieR)
    show_tqdm : bool
        Show progress bars
    verbose : bool
        Print per-filter summaries and totals

    Returns
    -------
    pd.DataFrame
        Full dataframe with added columns:
        - failed_<filter_name>: bool, True if row failed that filter
        - failed_any: bool, True if row failed any filter
    """
    df_filtered = df.copy()
    n_start = len(df_filtered)

    filters = []

    if apply_evidence_strength:
        filters.append(("posterior_strength", filter_evidence_strength, {
            "min_bayes_factor": min_bayes_factor,
            "require_finite_local_bf": require_finite_local_bf,
            "show_tqdm": show_tqdm,
            "verbose": verbose,
        }))

    if apply_run_robustness:
        filters.append(("run_robustness", filter_run_robustness, {
            "min_run_count": min_run_count,
            "min_run_points": min_run_points,
            "min_run_cameras": min_run_cameras,
            "show_tqdm": show_tqdm,
            "verbose": verbose,
        }))

    if apply_morphology:
        filters.append(("morphology", filter_morphology, {
            "dip_morphology": dip_morphology,
            "jump_morphology": jump_morphology,
            "min_delta_bic": min_delta_bic,
            "show_tqdm": show_tqdm,
            "verbose": verbose,
        }))

    if apply_score:
        filters.append(("score", filter_score, {
            "min_score": min_score,
            "show_tqdm": show_tqdm,
            "verbose": verbose,
        }))

    if apply_periodicity_validation:
        filters.append(("periodicity", validate_periodicity, {
            "n_bootstrap": periodicity_n_bootstrap,
            "significance_level": periodicity_significance,
            "exclude_alias_periods": periodicity_exclude_aliases,
            "flag_only": periodicity_flag_only,
            "workers": periodicity_workers,
            "checkpoint_dir": periodicity_checkpoint_dir,
            "show_tqdm": show_tqdm,
            "verbose": verbose,
        }))

    if apply_gaia_ruwe_validation:
        filters.append(("gaia_ruwe", validate_gaia_ruwe, {
            "max_ruwe": gaia_max_ruwe,
            "flag_only": gaia_flag_only,
            "show_tqdm": show_tqdm,
            "verbose": verbose,
        }))

    if apply_periodic_catalog_validation:
        filters.append(("periodic_catalog", validate_periodic_catalog, {
            "max_sep_arcsec": periodic_catalog_max_sep,
            "flag_only": periodic_catalog_flag_only,
            "show_tqdm": show_tqdm,
            "verbose": verbose,
        }))

    # Apply filters and tag failures (all rows kept)
    total_steps = len(filters)
    if total_steps > 0:
        with tqdm(total=total_steps, desc="apply_post_filters", leave=True, disable=not show_tqdm) as pbar:
            for label, func, kwargs in filters:
                start = perf_counter()
                # Run filter on full dataframe to identify which rows pass
                df_passed = func(df_filtered, **kwargs)
                elapsed = perf_counter() - start

                # Determine which rows failed by comparing paths
                passed_paths = set(df_passed["path"].astype(str))
                failed_mask = ~df_filtered["path"].astype(str).isin(passed_paths)
                df_filtered[f"failed_{label}"] = failed_mask

                n_failed = int(failed_mask.sum())
                if verbose:
                    pbar.set_postfix_str(f"{label}: {n_failed}/{n_start} failed ({elapsed:.2f}s)")
                else:
                    pbar.set_postfix_str("")
                pbar.update(1)

    # Add summary column
    failed_cols = [c for c in df_filtered.columns if c.startswith("failed_")]
    if failed_cols:
        df_filtered["failed_any"] = df_filtered[failed_cols].any(axis=1)

    if show_tqdm and verbose:
        n_failed_any = int(df_filtered["failed_any"].sum()) if "failed_any" in df_filtered.columns else 0
        tqdm.write(f"\n[apply_post_filters] {n_failed_any}/{n_start} failed at least one filter")

    return df_filtered.reset_index(drop=True)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply post-filters to events.py results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  malca post_filter --input results.csv --output results_filtered.csv
  malca post_filter --input results.csv --output results_filtered.csv --min-bayes-factor 20
  malca post_filter --input results.csv --output results_filtered.csv --apply-periodicity-validation
  malca post_filter --input results.csv --output results_filtered.csv --skip-gaia-ruwe-validation --skip-periodic-catalog-validation
"""
    )

    # I/O
    parser.add_argument("--detect-run", type=Path, default=None,
                        help="Detect run directory (e.g., output/runs/20250121_143052). If specified, reads from <detect-run>/results/ and writes filtered results there.")
    parser.add_argument("--input", type=Path, default=None, help="Input CSV/Parquet from events.py (overrides --detect-run)")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV/Parquet path (overrides default location)")
    parser.add_argument("--index-file", type=Path, default=Path("input/asassn_index_masked_concat_cleaned_20250919_154524_brotli.parquet"),
                        help="ASAS-SN index file to join ra_deg/dec_deg coordinates")

    # Filter toggles (all enabled by default except morphology)
    parser.add_argument("--skip-evidence-strength", action="store_true", help="Skip evidence strength filter (Bayes factor threshold)")
    parser.add_argument("--skip-run-robustness", action="store_true", help="Skip run robustness filter")
    parser.add_argument("--apply-morphology", action="store_true", help="Apply morphology filter (off by default)")

    # Posterior strength parameters
    parser.add_argument("--min-bayes-factor", type=float, default=10.0,
                        help="Minimum Bayes factor for posterior strength filter (default: 10)")
    parser.add_argument("--allow-infinite-local-bf", action="store_true",
                        help="Allow infinite local BF (default: require finite)")

    # Run robustness parameters
    parser.add_argument("--min-run-count", type=int, default=1,
                        help="Minimum number of runs (default: 1)")
    parser.add_argument("--min-run-points", type=int, default=2,
                        help="Minimum points per run (default: 2)")
    parser.add_argument("--min-run-cameras", type=int, default=2,
                        help="Minimum cameras per run (default: 2)")

    # Morphology parameters
    parser.add_argument("--dip-morphology", type=str, default="gaussian",
                        choices=["gaussian", "paczynski"],
                        help="Required morphology for dips (default: gaussian)")
    parser.add_argument("--jump-morphology", type=str, default="paczynski",
                        choices=["gaussian", "paczynski"],
                        help="Required morphology for jumps (default: paczynski)")
    parser.add_argument("--min-delta-bic", type=float, default=10.0,
                        help="Minimum delta BIC for morphology filter (default: 10)")

    # Score filter parameters
    parser.add_argument("--apply-score-filter", action="store_true",
                        help="Apply event score filter (off by default)")
    parser.add_argument("--min-score", type=float, default=-3.0,
                        help="Minimum log10 event score (dipper_score) to keep (default: -3.0)")

    # Periodicity validation parameters
    parser.add_argument("--apply-periodicity-validation", action="store_true",
                        help="Apply bootstrap LSP periodicity validation (off by default)")
    parser.add_argument("--periodicity-n-bootstrap", type=int, default=1000,
                        help="Number of bootstrap iterations (default: 1000)")
    parser.add_argument("--periodicity-significance", type=float, default=0.01,
                        help="Significance threshold (default: 0.01)")
    parser.add_argument("--periodicity-no-exclude-aliases", action="store_true",
                        help="Do not exclude alias periods (1d, 29.53d, etc.)")
    parser.add_argument("--periodicity-reject", action="store_true",
                        help="Reject periodic candidates (default: flag only)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers for LSP validation (default: 1)")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Directory for checkpoints (enables resume on restart)")


    # Gaia RUWE validation parameters
    parser.add_argument("--skip-gaia-ruwe-validation", action="store_true",
                        help="Skip Gaia RUWE validation (on by default, queries Gaia TAP)")
    parser.add_argument("--gaia-max-ruwe", type=float, default=1.4,
                        help="Maximum RUWE to keep (default: 1.4)")
    parser.add_argument("--gaia-reject", action="store_true",
                        help="Reject high RUWE sources (default: flag only)")

    # Periodic catalog validation parameters
    parser.add_argument("--skip-periodic-catalog-validation", action="store_true",
                        help="Skip periodic catalog crossmatch (on by default, fetches Chen+2020 from VizieR)")
    parser.add_argument("--periodic-catalog-max-sep", type=float, default=3.0,
                        help="Maximum separation in arcsec for catalog match (default: 3.0)")
    parser.add_argument("--periodic-catalog-reject", action="store_true",
                        help="Reject catalog matches (default: flag only)")

    # General options
    parser.add_argument("--no-tqdm", action="store_true", help="Disable progress bars")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print per-filter summaries (default: off)")

    args = parser.parse_args()

    # Determine input path
    if args.input:
        input_path = args.input.expanduser()
    elif args.detect_run:
        detect_run = args.detect_run.expanduser()
        results_dir = detect_run / "results"
        # Look for events results file in the detect run directory
        candidates = list(results_dir.glob("*events_results.csv")) + list(results_dir.glob("*events_results.parquet"))
        if not candidates:
            raise FileNotFoundError(f"No events results file found in {results_dir}")
        if len(candidates) > 1:
            print(f"Warning: Multiple results files found, using: {candidates[0]}")
        input_path = candidates[0]
    else:
        raise ValueError("Must specify either --input or --detect-run")

    # Load input
    if input_path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    print(f"Loaded {len(df)} rows from {input_path}")

    # Join coordinates from index file
    index_path = args.index_file.expanduser()
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    if index_path.suffix.lower() in (".parquet", ".pq"):
        index_df = pd.read_parquet(index_path)
    else:
        index_df = pd.read_csv(index_path)

    # Determine join column (asas_sn_id or path stem)
    if "asas_sn_id" in df.columns and "asas_sn_id" in index_df.columns:
        join_col = "asas_sn_id"
        # Ensure same type
        df[join_col] = df[join_col].astype(str)
        index_df[join_col] = index_df[join_col].astype(str)
    elif "path" in df.columns:
        # Extract ID from path (as string)
        df["_join_id"] = df["path"].apply(lambda p: Path(p).stem).astype(str)
        if "asas_sn_id" in index_df.columns:
            index_df["_join_id"] = index_df["asas_sn_id"].astype(str)
        join_col = "_join_id"
    else:
        raise ValueError("Cannot determine join column between events results and index file")

    # Join gaia_id, ra_deg, dec_deg from index
    join_cols = ["gaia_id", "ra_deg", "dec_deg"]
    available_cols = [c for c in join_cols if c in index_df.columns]
    if "gaia_id" not in available_cols:
        raise ValueError(f"Index file missing gaia_id column")

    df = df.merge(
        index_df[[join_col] + available_cols].drop_duplicates(subset=[join_col]),
        on=join_col,
        how="left"
    )
    if "_join_id" in df.columns:
        df = df.drop(columns=["_join_id"])
    print(f"Joined {len(available_cols)} columns ({', '.join(available_cols)}) from {index_path}")

    # Determine output path
    if args.output:
        output_path = args.output.expanduser()
    elif args.detect_run:
        detect_run = args.detect_run.expanduser()
        results_dir = detect_run / "results"
        # Create filtered filename based on input filename
        base_name = input_path.stem.replace("_results", "").replace("events", "")
        if base_name:
            filtered_name = f"{base_name}_events_results_filtered{input_path.suffix}"
        else:
            filtered_name = f"events_results_filtered{input_path.suffix}"
        output_path = results_dir / filtered_name
    else:
        # Fallback: same directory as input
        output_path = input_path.parent / f"{input_path.stem}_filtered{input_path.suffix}"

    # Apply filters
    df_filtered = apply_post_filters(
        df,
        # Filter toggles
        apply_evidence_strength=not args.skip_evidence_strength,
        apply_run_robustness=not args.skip_run_robustness,
        apply_morphology=args.apply_morphology,
        # Posterior strength
        min_bayes_factor=args.min_bayes_factor,
        require_finite_local_bf=not args.allow_infinite_local_bf,
        # Run robustness
        min_run_count=args.min_run_count,
        min_run_points=args.min_run_points,
        min_run_cameras=args.min_run_cameras,
        # Morphology
        dip_morphology=args.dip_morphology,
        jump_morphology=args.jump_morphology,
        min_delta_bic=args.min_delta_bic,
        # Score
        apply_score=args.apply_score_filter,
        min_score=args.min_score,
        # Periodicity validation
        apply_periodicity_validation=args.apply_periodicity_validation,
        periodicity_n_bootstrap=args.periodicity_n_bootstrap,
        periodicity_significance=args.periodicity_significance,
        periodicity_exclude_aliases=not args.periodicity_no_exclude_aliases,
        periodicity_flag_only=not args.periodicity_reject,
        periodicity_workers=args.workers,
        periodicity_checkpoint_dir=args.checkpoint_dir.expanduser() if args.checkpoint_dir else (detect_run / "checkpoints" if args.detect_run and args.apply_periodicity_validation else None),
        # Gaia RUWE validation
        apply_gaia_ruwe_validation=not args.skip_gaia_ruwe_validation,
        gaia_max_ruwe=args.gaia_max_ruwe,
        gaia_flag_only=not args.gaia_reject,
        # Periodic catalog validation
        apply_periodic_catalog_validation=not args.skip_periodic_catalog_validation,
        periodic_catalog_max_sep=args.periodic_catalog_max_sep,
        periodic_catalog_flag_only=not args.periodic_catalog_reject,
        # General
        show_tqdm=not args.no_tqdm,
        verbose=args.verbose,
    )

    # Generate filter log with comprehensive statistics
    if args.detect_run:
        try:
            import json
            import sys
            import shlex
            from datetime import datetime

            detect_run = args.detect_run.expanduser()
            filter_log_file = detect_run / "filter_log.json"

            orig_argv = getattr(sys, "orig_argv", None)
            cmd = shlex.join(orig_argv) if orig_argv else shlex.join([sys.executable] + sys.argv)

            filter_log = {
                "timestamp": datetime.now().isoformat(),
                "command": cmd,
                "input_file": str(input_path),
                "output_file": str(output_path),
                "filter_params": {
                    "apply_evidence_strength": not args.skip_evidence_strength,
                    "apply_run_robustness": not args.skip_run_robustness,
                    "apply_morphology": args.apply_morphology,
                    "apply_score": args.apply_score_filter,
                    "apply_periodicity_validation": args.apply_periodicity_validation,
                    "periodicity_reject": args.periodicity_reject if args.apply_periodicity_validation else None,
                    "apply_gaia_ruwe_validation": not args.skip_gaia_ruwe_validation,
                    "apply_periodic_catalog_validation": not args.skip_periodic_catalog_validation,
                    "min_bayes_factor": args.min_bayes_factor,
                    "require_finite_local_bf": not args.allow_infinite_local_bf,
                    "min_run_count": args.min_run_count,
                    "min_run_points": args.min_run_points,
                    "min_run_cameras": args.min_run_cameras,
                    "dip_morphology": args.dip_morphology if args.apply_morphology else None,
                    "jump_morphology": args.jump_morphology if args.apply_morphology else None,
                    "min_delta_bic": args.min_delta_bic if args.apply_morphology else None,
                    "min_score": args.min_score if args.apply_score_filter else None,
                    "gaia_max_ruwe": args.gaia_max_ruwe if not args.skip_gaia_ruwe_validation else None,
                    "gaia_reject": args.gaia_reject if not args.skip_gaia_ruwe_validation else None,
                    "periodic_catalog_max_sep": args.periodic_catalog_max_sep if not args.skip_periodic_catalog_validation else None,
                    "periodic_catalog_reject": args.periodic_catalog_reject if not args.skip_periodic_catalog_validation else None,
                },
                "results": {
                    "total_rows": len(df_filtered),
                    "passed_all": int((~df_filtered.get("failed_any", pd.Series(False))).sum()),
                    "failed_any": int(df_filtered.get("failed_any", pd.Series(False)).sum()),
                    "per_filter_failures": {
                        col: int(df_filtered[col].sum())
                        for col in df_filtered.columns
                        if col.startswith("failed_") and col != "failed_any"
                    },
                },
            }

            with open(filter_log_file, "w") as f:
                json.dump(filter_log, f, indent=2, default=str)

            if args.verbose:
                print(f"Filter log saved to {filter_log_file}")

        except Exception as e:
            if args.verbose:
                print(f"Warning: could not write filter log: {e}")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() in (".parquet", ".pq"):
        df_filtered.to_parquet(output_path, index=False)
    else:
        df_filtered.to_csv(output_path, index=False)

    n_failed = int(df_filtered["failed_any"].sum()) if "failed_any" in df_filtered.columns else 0
    n_passed = len(df_filtered) - n_failed
    print(f"\nWrote {len(df_filtered)} rows to {output_path}")
    print(f"Passed all filters: {n_passed}/{len(df_filtered)} ({n_passed/len(df_filtered)*100:.1f}%)")

    # Print per-filter failure counts
    failed_cols = [c for c in df_filtered.columns if c.startswith("failed_") and c != "failed_any"]
    if failed_cols:
        print("\nPer-filter failures:")
        for col in failed_cols:
            n = int(df_filtered[col].sum())
            print(f"  {col}: {n}/{len(df_filtered)} ({n/len(df_filtered)*100:.1f}%)")


if __name__ == "__main__":
    main()
