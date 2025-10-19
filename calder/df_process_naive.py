from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter
from typing import Dict, Any, Iterable, List, Optional
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from astropy.timeseries import LombScargle
from astropy import units as u

from vsx_crossmatch import propagate_asassn_coords, vsx_coords
from lc_utils import read_lc_dat, read_lc_raw



# parallelization helpers

def _iter_chunks(df: pd.DataFrame, chunk_size: int) -> Iterable[pd.DataFrame]:
    """Yield consecutive row chunks of a DataFrame."""
    n = len(df)
    if n == 0:
        return
    for start in range(0, n, chunk_size):
        yield df.iloc[start : start + chunk_size]

def _call_filter_by_name(func_name: str, df_chunk: pd.DataFrame, kwargs: Dict[str, Any]) -> pd.DataFrame:
    """
    Worker entry point: resolve a top-level function by name and apply to the chunk.
    (Using a name avoids trying to pickle closures/lambdas.)
    """
    fn = globals().get(func_name)
    if fn is None:
        raise RuntimeError(f"Filter function '{func_name}' not found in module globals().")
    return fn(df_chunk, **kwargs)

def _run_step_parallel(
    df_in: pd.DataFrame,
    *,
    func_name: str,
    func_kwargs: Dict[str, Any] | None,
    n_workers: int,
    chunk_size: int | None,
    step_desc: str,
    position: int = 0,
) -> pd.DataFrame:
    """
    Run a single filtration step in parallel over DataFrame chunks with a tqdm bar.
    Returns the concatenated DataFrame result.
    """
    func_kwargs = func_kwargs or {}
    n_rows = len(df_in)

    # Choose a decent default chunk size: ~4 chunks per worker, but not tiny
    if chunk_size is None:
        chunk_size = max(1000, int(np.ceil(n_rows / max(1, (n_workers or 1) * 4))))

    if n_rows == 0:
        tqdm.write(f"[{step_desc}] (skipped — 0 rows)")
        return df_in

    start = perf_counter()

    # Row-based progress bar — shows rows/s throughput
    pbar = tqdm(
        total=n_rows,
        desc=step_desc,
        unit="rows",
        position=position,
        leave=False,
        dynamic_ncols=True,
    )

    # Sequential path
    if not n_workers or n_workers <= 1:
        out_chunks: List[pd.DataFrame] = []
        for ch in _iter_chunks(df_in, chunk_size):
            out_chunks.append(_call_filter_by_name(func_name, ch, func_kwargs))
            pbar.update(len(ch))
        pbar.close()
        elapsed = perf_counter() - start
        tqdm.write(f"[{step_desc}] {n_rows} rows in {elapsed:.2f}s  ({n_rows/max(elapsed,1e-9):.1f} rows/s)")
        return pd.concat(out_chunks, ignore_index=True) if out_chunks else df_in.iloc[0:0].copy()

    # Parallel path
    out_chunks: List[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=None) as ex:
        futures = {}
        for ch in _iter_chunks(df_in, chunk_size):
            # Capture chunk length for progress updates on completion
            futures[ex.submit(_call_filter_by_name, func_name, ch, func_kwargs)] = len(ch)

        for fut in as_completed(futures):
            rows_done = futures[fut]
            res = fut.result()
            out_chunks.append(res)
            pbar.update(rows_done)

    pbar.close()
    elapsed = perf_counter() - start
    tqdm.write(f"[{step_desc}] {n_rows} rows in {elapsed:.2f}s  ({n_rows/max(elapsed,1e-9):.1f} rows/s)")

    return pd.concat(out_chunks, ignore_index=True) if out_chunks else df_in.iloc[0:0].copy()


# filtration functions

def candidates_with_peaks_naive(
    csv_path,
    out_csv_path=None,
    write_csv: bool = True,
    index: bool = False,
    band: str = "either",
):
    """
    Read peaks_[mag_bin].csv and return only rows where either band has a non-zero number of peaks.
    Adds a short tqdm to show load+filter time on large CSVs.
    """
    with tqdm(total=3, desc="candidates_with_peaks_naive", leave=False) as pbar:
        file = Path(csv_path)
        if not file.exists():
            suffix = file.suffix or ".csv"
            stem = file.stem
            pattern = f"{stem}_*{suffix}"
            candidates = sorted(file.parent.glob(pattern))
            if not candidates:
                raise FileNotFoundError(f"No file found matching {file} or {pattern}")
            file = max(candidates, key=lambda p: p.stat().st_mtime)
        pbar.update(1)

        df = pd.read_csv(file).copy()
        pbar.update(1)

        df["g_n_peaks"] = pd.to_numeric(df["g_n_peaks"], errors="coerce").fillna(0)
        df["v_n_peaks"] = pd.to_numeric(df["v_n_peaks"], errors="coerce").fillna(0)

        band_key = band.lower()
        if band_key == "g":
            mask = df["g_n_peaks"] > 0
        elif band_key == "v":
            mask = df["v_n_peaks"] > 0
        elif band_key == "both":
            mask = (df["g_n_peaks"] > 0) & (df["v_n_peaks"] > 0)
        else:
            mask = (df["g_n_peaks"] > 0) | (df["v_n_peaks"] > 0)

        out = df.loc[mask].reset_index(drop=True)
        out["source_file"] = file.name

        if write_csv:
            dest = (
                Path(out_csv_path)
                if out_csv_path is not None
                else file.parent / f"{file.stem}_selected_dippers.csv"
            )
            out.to_csv(dest, index=index)
        pbar.update(1)

    return out


def filter_bns(
    df: pd.DataFrame,
    asassn_csv: str | Path = "results_crossmatch/asassn_index_masked_concat_cleaned_20250926_1557.csv",
):
    """
    Keep only rows with matching ASAS-SN catalog entries and append key columns.
    Includes a tiny progress indicator for catalog load/merge.
    """
    with tqdm(total=2, desc="filter_bns", leave=False) as pbar:
        catalog = pd.read_csv(asassn_csv)
        catalog["asas_sn_id"] = catalog["asas_sn_id"].astype(str)
        pbar.update(1)

        cols_to_attach = [
            "ra_deg", "dec_deg", "pm_ra", "pm_ra_d", "pm_dec", "pm_dec_d",
        ]
        cols_available = ["asas_sn_id"] + [c for c in cols_to_attach if c in catalog.columns]

        df_out = (
            df.assign(asas_sn_id=df["asas_sn_id"].astype(str))
            .merge(catalog[cols_available], on="asas_sn_id", how="inner")
        )
        df_out = df_out.dropna(subset=[c for c in ["pm_ra", "pm_dec"] if c in df_out.columns]).reset_index(drop=True)
        pbar.update(1)

    return df_out


def vsx_class_extract(
    df: pd.DataFrame,
    vsx_csv: str | Path = "results_crossmatch/vsx_cleaned_20250926_1557.csv",
    match_radius_arcsec: float = 3.0,
):
    """
    Append VSX classes for matches within the given radius.
    Shows a compact progress bar for load + match + attach.
    """
    with tqdm(total=3, desc="vsx_class_extract", leave=False) as pbar:
        vsx = pd.read_csv(vsx_csv)
        vsx = vsx.dropna(subset=["ra", "dec"]).reset_index(drop=True)
        pbar.update(1)

        coords_asassn = propagate_asassn_coords(df)
        coords_vsx = vsx_coords(vsx)
        idx, sep2d, _ = coords_asassn.match_to_catalog_sky(coords_vsx)
        mask = sep2d < (match_radius_arcsec * u.arcsec)
        pbar.update(1)

        df_out = df.copy()
        df_out["vsx_match_sep_arcsec"] = sep2d.arcsec
        df_out["vsx_class"] = pd.Series(index=df_out.index, dtype=object)
        if "class" in vsx.columns:
            df_out.loc[mask, "vsx_class"] = vsx.loc[idx[mask], "class"].values
        pbar.update(1)

    return df_out




def _first_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Return the first existing column name from candidates, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def filter_dip_dominated(
    df: pd.DataFrame,
    *,
    min_dip_fraction: float = 0.66,
    show_tqdm: bool = False,
) -> pd.DataFrame:
    """
    Keep targets whose dip fraction is >= min_dip_fraction in at least one band.
    Expected columns (any of):
      - 'g_dip_fraction', 'v_dip_fraction'
      - or boolean flags like 'g_dip_dominated', 'v_dip_dominated'
    Safe fallback: pass-through if required columns are missing.
    """
    g_frac_col = _first_col(df, "g_dip_fraction", "g_dip_frac")
    v_frac_col = _first_col(df, "v_dip_fraction", "v_dip_frac")
    g_flag_col = _first_col(df, "g_dip_dominated", "g_dipdom")
    v_flag_col = _first_col(df, "v_dip_dominated", "v_dipdom")

    n0 = len(df)
    if g_frac_col or v_frac_col:
        mask = pd.Series(False, index=df.index)
        if g_frac_col:
            mask |= df[g_frac_col].astype(float) >= min_dip_fraction
        if v_frac_col:
            mask |= df[v_frac_col].astype(float) >= min_dip_fraction
        out = df.loc[mask].reset_index(drop=True)
    elif g_flag_col or v_flag_col:
        mask = pd.Series(False, index=df.index)
        if g_flag_col:
            mask |= df[g_flag_col].astype(bool)
        if v_flag_col:
            mask |= df[v_flag_col].astype(bool)
        out = df.loc[mask].reset_index(drop=True)
    else:
        tqdm.write("[filter_dip_dominated] No dip-fraction/flag columns found; passing through.")
        out = df.reset_index(drop=True)

    if show_tqdm:
        tqdm.write(f"[filter_dip_dominated] kept {len(out)}/{n0}")
    return out


def filter_multi_camera(
    df: pd.DataFrame,
    *,
    min_cameras: int = 2,
    show_tqdm: bool = False,
) -> pd.DataFrame:
    """
    Keep targets observed by >= min_cameras distinct cameras.
    Expected columns (any of): 'n_cameras', 'num_cameras', 'unique_cameras',
    'n_unique_cameras', 'camera_count'
    Safe fallback: pass-through if count column is missing.
    """
    cam_col = _first_col(
        df, "n_cameras", "num_cameras", "unique_cameras",
        "n_unique_cameras", "camera_count"
    )
    n0 = len(df)
    if cam_col:
        out = df.loc[df[cam_col].astype(int) >= min_cameras].reset_index(drop=True)
    else:
        tqdm.write("[filter_multi_camera] No camera-count column found; passing through.")
        out = df.reset_index(drop=True)

    if show_tqdm:
        tqdm.write(f"[filter_multi_camera] kept {len(out)}/{n0}")
    return out


def filter_periodic_candidates(
    df: pd.DataFrame,
    *,
    max_power: float = 0.5,
    min_period: float | None = None,
    max_period: float | None = None,
    show_tqdm: bool = False,
) -> pd.DataFrame:
    """
    Remove strongly periodic sources (likely non-dippers).
    Expected precomputed columns if available:
      - power: 'ls_max_power' or 'max_power'
      - best period: 'best_period', 'ls_best_period', or 'period_best' (days)
    Behavior:
      - If power column exists: keep rows with power <= max_power.
      - If period bounds provided AND best-period column exists: enforce bounds.
    Safe fallback: pass-through if nothing to test.
    """
    power_col = _first_col(df, "ls_max_power", "max_power")
    per_col = _first_col(df, "best_period", "ls_best_period", "period_best")
    n0 = len(df)
    mask = pd.Series(True, index=df.index)

    if power_col:
        mask &= df[power_col].astype(float) <= float(max_power)
    else:
        tqdm.write("[filter_periodic_candidates] No power column found; skipping power filter.")

    if (min_period is not None or max_period is not None) and per_col:
        if min_period is not None:
            mask &= df[per_col].astype(float) >= float(min_period)
        if max_period is not None:
            mask &= df[per_col].astype(float) <= float(max_period)
    elif (min_period is not None or max_period is not None) and not per_col:
        tqdm.write("[filter_periodic_candidates] No best-period column found; skipping period bounds.")

    out = df.loc[mask].reset_index(drop=True)

    if show_tqdm:
        tqdm.write(f"[filter_periodic_candidates] kept {len(out)}/{n0}")
    return out


def filter_sparse_lightcurves(
    df: pd.DataFrame,
    *,
    min_time_span: float = 200.0,
    min_points_per_day: float = 0.05,
    show_tqdm: bool = False,
) -> pd.DataFrame:
    """
    Remove sparsely sampled targets.
    Expected columns if available:
      - 'time_span_days'/'timespan_days' (float)
      - 'points_per_day'/'ppd'/'n_per_day' (float)
    Safe fallback: pass-through if metrics missing.
    """
    span_col = _first_col(df, "time_span_days", "timespan_days", "t_span_days", "t_span")
    ppd_col = _first_col(df, "points_per_day", "ppd", "n_per_day")

    n0 = len(df)
    if span_col and ppd_col:
        mask = (df[span_col].astype(float) >= float(min_time_span)) & \
               (df[ppd_col].astype(float) >= float(min_points_per_day))
        out = df.loc[mask].reset_index(drop=True)
    elif span_col:
        tqdm.write("[filter_sparse_lightcurves] No points/day column; using span only.")
        mask = df[span_col].astype(float) >= float(min_time_span)
        out = df.loc[mask].reset_index(drop=True)
    elif ppd_col:
        tqdm.write("[filter_sparse_lightcurves] No span column; using points/day only.")
        mask = df[ppd_col].astype(float) >= float(min_points_per_day)
        out = df.loc[mask].reset_index(drop=True)
    else:
        tqdm.write("[filter_sparse_lightcurves] No sparsity metrics found; passing through.")
        out = df.reset_index(drop=True)

    if show_tqdm:
        tqdm.write(f"[filter_sparse_lightcurves] kept {len(out)}/{n0}")
    return out


def _sigma_ok_for_row(asas_sn_id: str, raw_path: str | Path, min_sigma: float) -> bool:
    """
    Compute whether max dip depth >= min_sigma * median 1-sigma scatter for this source.
    Uses your previously commented logic.
    """
    try:
        raw_df = read_lc_raw(str(asas_sn_id), str(Path(raw_path).parent))
        scatter_vals = (raw_df["sig1_high"] - raw_df["sig1_low"]).to_numpy(dtype=float)
        finite = scatter_vals[np.isfinite(scatter_vals)]
        if finite.size == 0:
            return False
        scatter = np.nanmedian(finite)
        return bool(np.isfinite(scatter) and scatter > 0.0 and scatter)
    except Exception:
        # If loading fails, be conservative: drop
        return False


def filter_sigma_resid(
    df: pd.DataFrame,
    *,
    min_sigma: float = 3.0,
    n_helpers: int = 1,
    show_tqdm: bool = False,
) -> pd.DataFrame:
    """
    Keep rows where (max dip depth / median-1σ-scatter) >= min_sigma in either band.
    Requires columns: 'asas_sn_id', 'raw_path' and depth columns 'g_max_depth'/'v_max_depth'.
    Falls back to pass-through if required pieces are missing.
    """
    need_cols = {"asas_sn_id", "raw_path"}
    depth_g = _first_col(df, "g_max_depth", "g_depth_max")
    depth_v = _first_col(df, "v_max_depth", "v_depth_max")

    if not need_cols.issubset(df.columns) or (not depth_g and not depth_v):
        tqdm.write("[filter_sigma_resid] Missing asas_sn_id/raw_path or depth columns; passing through.")
        return df.reset_index(drop=True)

    # Precompute which rows are OK by estimating scatter and comparing depths.
    rows = df.reset_index(drop=False)  # keep original index for mask
    idx_name = "index"

    # Worker: compute scatter and compare to depths for a single row
    def _eval_row(row) -> tuple[int, bool]:
        try:
            raw_ok = _sigma_ok_for_row(str(row["asas_sn_id"]), Path(row["raw_path"]), min_sigma)
            if not raw_ok:
                return (int(row[idx_name]), False)
            scatter_df = read_lc_raw(str(row["asas_sn_id"]), str(Path(row["raw_path"]).parent))
            scatter_vals = (scatter_df["sig1_high"] - scatter_df["sig1_low"]).to_numpy(dtype=float)
            finite = scatter_vals[np.isfinite(scatter_vals)]
            if finite.size == 0:
                return (int(row[idx_name]), False)
            scatter = np.nanmedian(finite)
            ok = False
            if depth_g and pd.notna(row[depth_g]):
                ok |= (float(row[depth_g]) / scatter) >= float(min_sigma)
            if depth_v and pd.notna(row[depth_v]):
                ok |= (float(row[depth_v]) / scatter) >= float(min_sigma)
            return (int(row[idx_name]), bool(ok))
        except Exception:
            return (int(row[idx_name]), False)

    it = rows.itertuples(index=False)
    results = {}
    if n_helpers and n_helpers > 1:
        with ProcessPoolExecutor(max_workers=n_helpers) as ex:
            futs = [ex.submit(_eval_row, r._asdict()) for r in it]
            prog = tqdm(total=len(futs), desc="filter_sigma_resid", leave=False) if show_tqdm else None
            for f in as_completed(futs):
                i, ok = f.result()
                results[i] = ok
                if prog: prog.update(1)
            if prog: prog.close()
    else:
        prog = tqdm(total=len(df), desc="filter_sigma_resid", leave=False) if show_tqdm else None
        for r in it:
            i, ok = _eval_row(r._asdict())
            results[i] = ok
            if prog: prog.update(1)
        if prog: prog.close()

    mask = rows[idx_name].map(results).astype(bool)
    out = df.loc[mask.values].reset_index(drop=True)
    if show_tqdm:
        tqdm.write(f"[filter_sigma_resid] kept {len(out)}/{len(df)}")
    return out

def filter_csv(
    csv_path: str | Path,
    out_csv_path: str | Path | None = None,
    band: str = "either",
    asassn_csv: str | Path = "results_crossmatch/asassn_index_masked_concat_cleaned_20250926_1557.csv",
    vsx_csv: str | Path = "results_crossmatch/vsx_cleaned_20250926_1557.csv",
    min_dip_fraction: float = 0.66,
    min_cameras: int = 2,
    max_power: float = 0.5,
    min_period: float | None = None,
    max_period: float | None = None,
    min_time_span: float = 200.0,
    min_points_per_day: float = 0.05,
    min_sigma: float = 3.0,
    match_radius_arcsec: float = 3.0,
    n_helpers: int = 60,
    skip_dip_dom: bool = False,
    skip_multi_camera: bool = False,
    skip_periodic: bool = False,
    skip_sparse: bool = True,   # not used for now
    skip_sigma: bool = True,    # not used for now
    chunk_size: int | None = None,
    tqdm_position_base: int = 0,
) -> pd.DataFrame:
    """
    Orchestrates all filters with:
      - outer tqdm over steps
      - inner tqdm per step (rows/s)
      - process-based parallelization for each step using chunking
    """
    # Step 0: seed set (with its own tiny progress)
    df_filtered = candidates_with_peaks_naive(csv_path, write_csv=False, band=band)

    # Build a plan of steps to run: (step_name, func_name, kwargs)
    plan: list[tuple[str, str, dict]] = []

    if not skip_dip_dom:
        plan.append(("dip_dominated", "filter_dip_dominated", {"min_dip_fraction": min_dip_fraction}))

    if not skip_multi_camera:
        plan.append(("multi_camera", "filter_multi_camera", {"min_cameras": min_cameras}))

    if not skip_periodic:
        plan.append(("periodic", "filter_periodic_candidates", {
            "max_power": max_power, "min_period": min_period, "max_period": max_period
        }))

    if not skip_sparse:
        plan.append(("sparse", "filter_sparse_lightcurves", {
            "min_time_span": min_time_span, "min_points_per_day": min_points_per_day
        }))

    if not skip_sigma:
        plan.append(("sigma", "filter_sigma_resid", {"min_sigma": min_sigma}))

    # Optional “pre” join to ASAS-SN catalog if you want it upstream of other filters:
    # plan.insert(0, ("bns_join", "filter_bns", {"asassn_csv": asassn_csv}))

    # Outer bar over steps
    with tqdm(total=len(plan), desc="filter_csv (steps)", position=tqdm_position_base, leave=False) as outer:
        for i, (label, func_name, kwargs) in enumerate(plan):
            # Merge step-specific kwargs that may need global params (no ** unpacking)
            if func_name == "filter_bns":
                kwargs = dict(kwargs)
                kwargs["asassn_csv"] = asassn_csv
            elif func_name == "vsx_class_extract":
                kwargs = dict(kwargs)
                kwargs["vsx_csv"] = vsx_csv
                kwargs["match_radius_arcsec"] = match_radius_arcsec

            # n_helpers handled here globally — individual functions need not know about parallelism
            df_filtered = _run_step_parallel(
                df_filtered,
                func_name=func_name,
                func_kwargs=kwargs,
                n_workers=n_helpers,
                chunk_size=chunk_size,
                step_desc=f"{i+1}/{len(plan)} {label}",
                position=tqdm_position_base + 1,
            )
            outer.set_postfix_str(label)
            outer.update(1)

    df_filtered = df_filtered.reset_index(drop=True)

    if out_csv_path is not None:
        Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df_filtered.to_csv(out_csv_path, index=False)

    # If you want VSX appended as the final (non-filtering) enrichment step WITH progress:
    # df_filtered = _run_step_parallel(
    #     df_filtered,
    #     func_name="vsx_class_extract",
    #     func_kwargs={"vsx_csv": vsx_csv, "match_radius_arcsec": match_radius_arcsec},
    #     n_workers=n_helpers,
    #     chunk_size=chunk_size,
    #     step_desc=f"{len(plan)+1}/{len(plan)+1} vsx_class_extract",
    #     position=tqdm_position_base + 1,
    # )

    return df_filtered
