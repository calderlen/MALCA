from __future__ import annotations

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter
from typing import Dict, Any, Iterable, List

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from astropy.timeseries import LombScargle
from astropy import units as u

from vsx_crossmatch import propagate_asassn_coords, vsx_coords

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


# ---------------------------
# Your functions (with tqdm)
# ---------------------------

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


# ---------------------------------------------------
# filter_csv with nested tqdm + per-step parallelism
# ---------------------------------------------------

def filter_csv(
    csv_path: str | Path,
    *,
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
    skip_sparse: bool = False,
    skip_sigma: bool = False,
    # new knobs for parallel chunking and tqdm layout
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
    # (uncomment if desired as an explicit step)
    # plan.insert(0, ("bns_join", "filter_bns", {"asassn_csv": asassn_csv}))

    # Outer bar over steps
    with tqdm(total=len(plan), desc="filter_csv (steps)", position=tqdm_position_base, leave=False) as outer:
        for i, (label, func_name, kwargs) in enumerate(plan):
            # Merge step-specific kwargs that may need global params:
            if func_name == "filter_bns":
                kwargs = {**kwargs, "asassn_csv": asassn_csv}
            elif func_name == "vsx_class_extract":
                kwargs = {**kwargs, "vsx_csv": vsx_csv, "match_radius_arcsec": match_radius_arcsec}
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
