from __future__ import annotations
import argparse
import fnmatch
import re
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter
from typing import Dict, Any, Optional, Iterable
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from astropy.timeseries import LombScargle
from astropy import units as u

from vsx_crossmatch import propagate_asassn_coords, vsx_coords
from lc_utils import read_lc_dat, read_lc_raw



def _call_filter_by_name(func_name: str, df_in: pd.DataFrame, kwargs: Dict[str, Any]) -> pd.DataFrame:
    fn = globals()[func_name]
    return fn(df_in, **kwargs)

def _filter_candidates_chunk(df_chunk: pd.DataFrame, band_key: str) -> pd.DataFrame:
    if df_chunk.empty:
        return df_chunk.iloc[0:0].copy()

    if band_key == "g":
        mask = df_chunk["g_n_peaks"] > 0
    elif band_key == "v":
        mask = df_chunk["v_n_peaks"] > 0
    elif band_key == "both":
        mask = (df_chunk["g_n_peaks"] > 0) & (df_chunk["v_n_peaks"] > 0)
    else:
        mask = (df_chunk["g_n_peaks"] > 0) | (df_chunk["v_n_peaks"] > 0)

    return df_chunk.loc[mask].copy()

def _run_step_sequential(
    df_in: pd.DataFrame,
    *,
    func_name: str,
    func_kwargs: Dict[str, Any] | None,
    step_desc: str,
    position: int = 0,
) -> pd.DataFrame:
    func_kwargs = func_kwargs or {}
    start = perf_counter()
    n_in = len(df_in)

    with tqdm(total=1, desc=step_desc, position=position, leave=False, dynamic_ncols=True) as pbar:
        df_out = _call_filter_by_name(func_name, df_in, func_kwargs)
        pbar.update(1)

    elapsed = perf_counter() - start
    tqdm.write(
        f"[{step_desc}] {n_in} → {len(df_out)} rows in {elapsed:.2f}s"
    )
    return df_out

def _first_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _log_rejections(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    filter_name: str,
    log_csv: str | Path | None,
) -> None:
    if log_csv is None:
        return

    before_ids = set(df_before["asas_sn_id"].astype(str))
    after_ids = set(df_after["asas_sn_id"].astype(str)) if "asas_sn_id" in df_after.columns else set()
    rejected = sorted(before_ids - after_ids)
    if not rejected:
        return

    log_path = Path(log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df_log = pd.DataFrame({"asas_sn_id": rejected, "filter": filter_name})
    df_log.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)


def candidates_with_peaks_naive(
    csv_path,
    out_csv_path=None,
    write_csv: bool = True,
    index: bool = False,
    band: str = "either",
    *,
    n_workers: int = 1,
) -> pd.DataFrame:
    file = Path(csv_path)

    df = pd.read_csv(file).copy()
    df["g_n_peaks"] = pd.to_numeric(df["g_n_peaks"], errors="coerce").fillna(0)
    df["v_n_peaks"] = pd.to_numeric(df["v_n_peaks"], errors="coerce").fillna(0)

    band_key = band.lower()
    n_rows = len(df)

    if n_workers and n_workers > 1 and n_rows > 0:
        splits = [chunk for chunk in np.array_split(df, min(n_workers, n_rows)) if len(chunk)]
        out_chunks: list[pd.DataFrame] = []
        with tqdm(
            total=n_rows,
            desc="candidates_with_peaks_naive",
            unit="rows",
            leave=False,
            dynamic_ncols=True,
        ) as pbar:
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                futures = {
                    ex.submit(_filter_candidates_chunk, chunk, band_key): len(chunk)
                    for chunk in splits
                }
                for fut in as_completed(futures):
                    out_chunks.append(fut.result())
                    pbar.update(futures[fut])
        out = pd.concat(out_chunks, ignore_index=True) if out_chunks else df.iloc[0:0].copy()
    else:
        with tqdm(total=3, desc="candidates_with_peaks_naive", leave=False) as pbar:
            pbar.update(1)
            if band_key == "g":
                mask = df["g_n_peaks"] > 0
            elif band_key == "v":
                mask = df["v_n_peaks"] > 0
            elif band_key == "both":
                mask = (df["g_n_peaks"] > 0) & (df["v_n_peaks"] > 0)
            else:
                mask = (df["g_n_peaks"] > 0) | (df["v_n_peaks"] > 0)
            pbar.update(1)
            out = df.loc[mask].reset_index(drop=True)
            pbar.update(1)

    out["source_file"] = file.name

    if write_csv:
        dest = (
            Path(out_csv_path)
            if out_csv_path is not None
            else file.parent / f"{file.stem}_selected_dippers.csv"
        )
        out.to_csv(dest, index=index)

    return out


def candidates_with_peaks_biweight(
    csv_path,
    out_csv_path=None,
    write_csv: bool = True,
    index: bool = False,
    band: str = "either",
    *,
    n_workers: int = 1,
) -> pd.DataFrame:
    # Same selection logic as naive; operates on biweight peak CSVs.
    return candidates_with_peaks_naive(
        csv_path,
        out_csv_path=out_csv_path,
        write_csv=write_csv,
        index=index,
        band=band,
        n_workers=n_workers,
    )


def filter_bns(
    df: pd.DataFrame,
    asassn_csv: str | Path = "results_crossmatch/asassn_index_masked_concat_cleaned_20250926_1557.csv",
    rejected_log_csv: str | Path | None = None,
):
    with tqdm(total=2, desc="filter_bns", leave=False) as pbar:
        catalog = pd.read_csv(asassn_csv)
        catalog["asas_sn_id"] = catalog["asas_sn_id"].astype(str)
        pbar.update(1)

        cols_to_attach = [
            "ra_deg", "dec_deg", "pm_ra", "pm_ra_d", "pm_dec", "pm_dec_d",
        ]
        cols_available = ["asas_sn_id"] + [c for c in cols_to_attach if c in catalog.columns]

        df_ids = df.assign(asas_sn_id=df["asas_sn_id"].astype(str))
        df_out = (
            df_ids
            .merge(catalog[cols_available], on="asas_sn_id", how="inner")
        )
        df_out = df_out.dropna(subset=[c for c in ["pm_ra", "pm_dec"] if c in df_out.columns]).reset_index(drop=True)
        pbar.update(1)

    _log_rejections(df_ids, df_out, "filter_bns", rejected_log_csv)

    return df_out


def vsx_class_extract(
    df: pd.DataFrame,
    vsx_csv: str | Path = "results_crossmatch/vsx_cleaned_20250926_1557.csv",
    match_radius_arcsec: float = 3.0,
):
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

def filter_dip_dominated(
    df: pd.DataFrame,
    *,
    min_dip_fraction: float = 0.66,
    show_tqdm: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    g_frac_col = _first_col(df, "g_dip_fraction", "g_dip_frac")
    v_frac_col = _first_col(df, "v_dip_fraction", "v_dip_frac")
    g_flag_col = _first_col(df, "g_dip_dominated", "g_dipdom")
    v_flag_col = _first_col(df, "v_dip_dominated", "v_dipdom")

    n0 = len(df)
    pbar = tqdm(total=2, desc="filter_dip_dominated", leave=False) if show_tqdm else None
    mask = pd.Series(False, index=df.index)
    if g_frac_col:
        mask |= df[g_frac_col].astype(float) >= min_dip_fraction
    if v_frac_col:
        mask |= df[v_frac_col].astype(float) >= min_dip_fraction
    if g_flag_col:
        mask |= df[g_flag_col].astype(bool)
    if v_flag_col:
        mask |= df[v_flag_col].astype(bool)
    if not mask.any():
        raise KeyError("No dip-fraction or dip-dominated columns available for filtering.")
    out = df.loc[mask].reset_index(drop=True)
    if pbar:
        pbar.update(1)

    if show_tqdm:
        tqdm.write(f"[filter_dip_dominated] kept {len(out)}/{n0}")
    _log_rejections(df, out, "filter_dip_dominated", rejected_log_csv)
    if pbar:
        pbar.update(1)
        pbar.close()
    return out


def filter_multi_camera(
    df: pd.DataFrame,
    *,
    min_cameras: int = 2,
    show_tqdm: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    cam_col = _first_col(
        df, "n_cameras", "num_cameras", "unique_cameras",
        "n_unique_cameras", "camera_count"
    )
    n0 = len(df)
    pbar = tqdm(total=2, desc="filter_multi_camera", leave=False) if show_tqdm else None
    if cam_col:
        out = df.loc[df[cam_col].astype(int) >= min_cameras].reset_index(drop=True)
        if pbar:
            pbar.update(1)
    else:
        tqdm.write("[filter_multi_camera] No camera-count column found; passing through.")
        out = df.reset_index(drop=True)
        if pbar:
            pbar.update(1)

    if show_tqdm:
        tqdm.write(f"[filter_multi_camera] kept {len(out)}/{n0}")
    _log_rejections(df, out, "filter_multi_camera", rejected_log_csv)
    if pbar:
        pbar.update(1)
        pbar.close()
    return out


def filter_periodic_candidates(
    df: pd.DataFrame,
    *,
    max_power: float = 0.5,
    min_period: float | None = None,
    max_period: float | None = None,
    show_tqdm: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    power_col = _first_col(df, "ls_max_power", "max_power")
    per_col = _first_col(df, "best_period", "ls_best_period", "period_best")
    n0 = len(df)
    mask = pd.Series(True, index=df.index)
    pbar = tqdm(total=3, desc="filter_periodic_candidates", leave=False) if show_tqdm else None

    if power_col:
        mask &= df[power_col].astype(float) <= float(max_power)
        if pbar:
            pbar.update(1)
    else:
        tqdm.write("[filter_periodic_candidates] No power column found; skipping power filter.")
        if pbar:
            pbar.update(1)

    if (min_period is not None or max_period is not None) and per_col:
        if min_period is not None:
            mask &= df[per_col].astype(float) >= float(min_period)
        if max_period is not None:
            mask &= df[per_col].astype(float) <= float(max_period)
        if pbar:
            pbar.update(1)
    elif (min_period is not None or max_period is not None) and not per_col:
        tqdm.write("[filter_periodic_candidates] No best-period column found; skipping period bounds.")
        if pbar:
            pbar.update(1)
    else:
        if pbar:
            pbar.update(1)

    out = df.loc[mask].reset_index(drop=True)

    if show_tqdm:
        tqdm.write(f"[filter_periodic_candidates] kept {len(out)}/{n0}")
    _log_rejections(df, out, "filter_periodic_candidates", rejected_log_csv)
    if pbar:
        pbar.update(1)
        pbar.close()
    return out


def filter_sparse_lightcurves(
    df: pd.DataFrame,
    *,
    min_time_span: float = 200.0,
    min_points_per_day: float = 0.05,
    show_tqdm: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    span_col = _first_col(df, "time_span_days", "timespan_days", "t_span_days", "t_span")
    ppd_col = _first_col(df, "points_per_day", "ppd", "n_per_day")

    n0 = len(df)
    pbar = tqdm(total=3, desc="filter_sparse_lightcurves", leave=False) if show_tqdm else None
    if pbar:
        pbar.update(1)

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

    if pbar:
        pbar.update(1)

    if show_tqdm:
        tqdm.write(f"[filter_sparse_lightcurves] kept {len(out)}/{n0}")
    _log_rejections(df, out, "filter_sparse_lightcurves", rejected_log_csv)
    if pbar:
        pbar.update(1)
        pbar.close()
    return out


def box_filter(
    df: pd.DataFrame,
    *,
    min_dips: int = 1,
    max_dips: int = 9,
    max_peaks_per_time: float = 0.015,
    max_std_mag: float = 0.15,
    dip_cols: tuple[str, ...] = (
        "Number of dips",
        "number_of_dips",
        "n_dips",
        "n_dip_runs",
        "g_n_dip_runs",
        "v_n_dip_runs",
    ),
    peaks_cols: tuple[str, ...] = (
        "Peaks per time",
        "peaks_per_time",
        "peaks_per_day",
        "g_peaks_per_time",
        "v_peaks_per_time",
    ),
    std_cols: tuple[str, ...] = (
        "Standard Deviation of Magnitude",
        "standard_deviation_of_magnitude",
        "std_mag",
        "mag_std",
        "g_std_mag",
        "v_std_mag",
    ),
    show_tqdm: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    dip_col = _first_col(df, *dip_cols)
    peaks_col = _first_col(df, *peaks_cols)
    std_col = _first_col(df, *std_cols)

    missing: list[str] = []
    if dip_col is None:
        missing.append("dip-count column")
    if peaks_col is None:
        missing.append("peaks-per-time column")
    if std_col is None:
        missing.append("std-mag column")

    if missing:
        tqdm.write(f"[box_filter] Missing {', '.join(missing)}; passing through.")
        return df.reset_index(drop=True)

    dips = pd.to_numeric(df[dip_col], errors="coerce")
    peaks = pd.to_numeric(df[peaks_col], errors="coerce")
    stds = pd.to_numeric(df[std_col], errors="coerce")

    mask = (
        dips.notna()
        & peaks.notna()
        & stds.notna()
        & (dips >= int(min_dips))
        & (dips <= int(max_dips))
        & (peaks <= float(max_peaks_per_time))
        & (stds <= float(max_std_mag))
    )

    out = df.loc[mask].reset_index(drop=True)
    if show_tqdm:
        tqdm.write(
            f"[box_filter] kept {len(out)}/{len(df)} "
            f"(dips in [{min_dips}, {max_dips}], peaks/time <= {max_peaks_per_time}, std <= {max_std_mag})"
        )
    _log_rejections(df, out, "box_filter", rejected_log_csv)
    return out


def _sigma_ok_for_row(asas_sn_id: str, raw_path: str | Path, min_sigma: float) -> bool:
    try:
        raw_df = read_lc_raw(str(asas_sn_id), str(Path(raw_path).parent))
        scatter_vals = (raw_df["sig1_high"] - raw_df["sig1_low"]).to_numpy(dtype=float)
        finite = scatter_vals[np.isfinite(scatter_vals)]
        if finite.size == 0:
            return False
        scatter = np.nanmedian(finite)
        return bool(np.isfinite(scatter) and scatter > 0.0 and scatter)
    except Exception:
        return False


def filter_sigma_resid(
    df: pd.DataFrame,
    *,
    min_sigma: float = 3.0,
    show_tqdm: bool = False,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:
    need_cols = {"asas_sn_id", "raw_path"}
    depth_g = _first_col(df, "g_max_depth", "g_depth_max")
    depth_v = _first_col(df, "v_max_depth", "v_depth_max")

    if not need_cols.issubset(df.columns) or (not depth_g and not depth_v):
        tqdm.write("[filter_sigma_resid] Missing asas_sn_id/raw_path or depth columns; passing through.")
        return df.reset_index(drop=True)

    rows = df.reset_index(drop=False)  # keep original index for mask
    idx_name = "index"

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
    prog = tqdm(total=len(df), desc="filter_sigma_resid (rows)", leave=False) if show_tqdm else None
    for r in it:
        i, ok = _eval_row(r._asdict())
        results[i] = ok
        if prog:
            prog.update(1)
    if prog:
        prog.close()

    mask = rows[idx_name].map(results).astype(bool)
    out = df.loc[mask.values].reset_index(drop=True)
    if show_tqdm:
        tqdm.write(f"[filter_sigma_resid] kept {len(out)}/{len(df)}")
    _log_rejections(df, out, "filter_sigma_resid", rejected_log_csv)
    return out

def filter_csv(
    csv_path: str | Path,
    out_csv_path: str | Path | None = None,
    band: str = "either",
    peak_mode: str = "naive",
    asassn_csv: str | Path = "results_crossmatch/asassn_index_masked_concat_cleaned_20250926_1557.csv",
    vsx_csv: str | Path = "results_crossmatch/vsx_cleaned_20250926_1557.csv",
    min_dip_fraction: float = 0.0,
    min_cameras: int = 2,
    max_power: float = 0.5,
    min_period: float | None = None,
    max_period: float | None = None,
    min_time_span: float = 200.0,
    min_points_per_day: float = 0.05,
    apply_box: bool = False,
    box_min_dips: int = 1,
    box_max_dips: int = 9,
    box_max_peaks_per_time: float = 0.015,
    box_max_std_mag: float = 0.15,
    min_sigma: float = 3.0,
    match_radius_arcsec: float = 3.0,
    apply_bns: bool = True,
    apply_vsx_class: bool = True,
    apply_dip_dom: bool = True,
    apply_multi_camera: bool = True,
    apply_periodic: bool = True,
    apply_sparse: bool = True,
    apply_sigma: bool = True,
    seed_workers: int = 1,
    tqdm_position_base: int = 0,
    rejected_log_csv: str | Path | None = None,
) -> pd.DataFrame:

    if peak_mode == "biweight":
        df_filtered = candidates_with_peaks_biweight(
            csv_path,
            write_csv=False,
            band=band,
            n_workers=seed_workers,
        )
    else:
        df_filtered = candidates_with_peaks_naive(
            csv_path,
            write_csv=False,
            band=band,
            n_workers=seed_workers,
        )


    plan: list[tuple[str, str, dict]] = []

    if apply_dip_dom:
        plan.append(("dip_dominated", "filter_dip_dominated", {
            "min_dip_fraction": min_dip_fraction,
            "show_tqdm": True,
            "rejected_log_csv": rejected_log_csv,
        }))

    if apply_multi_camera:
        plan.append(("multi_camera", "filter_multi_camera", {
            "min_cameras": min_cameras,
            "show_tqdm": True,
            "rejected_log_csv": rejected_log_csv,
        }))

    if apply_box:
        plan.append(("box", "box_filter", {
            "min_dips": box_min_dips,
            "max_dips": box_max_dips,
            "max_peaks_per_time": box_max_peaks_per_time,
            "max_std_mag": box_max_std_mag,
            "show_tqdm": True,
            "rejected_log_csv": rejected_log_csv,
        }))

    if apply_sparse:
        plan.append(("sparse", "filter_sparse_lightcurves", {
            "min_time_span": min_time_span,
            "min_points_per_day": min_points_per_day,
            "show_tqdm": True,
            "rejected_log_csv": rejected_log_csv,
        }))

    if apply_bns:
        plan.append(("bns_join", "filter_bns", {
            "asassn_csv": asassn_csv,
            "rejected_log_csv": rejected_log_csv,
        }))

    if apply_sigma:
        plan.append(("sigma", "filter_sigma_resid", {
            "min_sigma": min_sigma,
            "show_tqdm": True,
            "rejected_log_csv": rejected_log_csv,
        }))

    if apply_vsx_class:
        plan.append(("vsx_class", "vsx_class_extract", {
            "vsx_csv": vsx_csv, "match_radius_arcsec": match_radius_arcsec
        }))

    if apply_periodic:
        plan.append(("periodic", "filter_periodic_candidates", {
            "max_power": max_power,
            "min_period": min_period,
            "max_period": max_period,
            "show_tqdm": True,
            "rejected_log_csv": rejected_log_csv,
        }))

    total_steps = len(plan)
    if total_steps:
        with tqdm(total=total_steps, desc="filter_csv (steps)", position=tqdm_position_base, leave=False) as outer:
            for i, (label, func_name, kwargs) in enumerate(plan):
                df_filtered = _run_step_sequential(
                    df_filtered,
                    func_name=func_name,
                    func_kwargs=kwargs,
                    step_desc=f"{i+1}/{total_steps} {label}",
                    position=tqdm_position_base + 1,
                )
                outer.set_postfix_str(label)
                outer.update(1)

    df_filtered = df_filtered.reset_index(drop=True)

    if out_csv_path is not None:
        Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df_filtered.to_csv(out_csv_path, index=False)

    return df_filtered



BIN_RE = re.compile(
    r"^peaks_(?P<low>\d+(?:_\d)?)_(?P<high>\d+(?:_\d)?)_(?P<ts>\d{8}_\d{6}[+-]\d{4})\.csv$"
)


def parse_bin_key(p: Path) -> tuple[str, str, str] | None:
    m = BIN_RE.match(p.name)
    if not m:
        return None
    return (m.group("low"), m.group("high"), m.group("ts"))


def latest_per_bin(files: list[Path]) -> list[Path]:
    best: dict[tuple[str, str], tuple[str, Path]] = {}
    for f in files:
        parsed = parse_bin_key(f)
        if not parsed:
            continue
        low, high, ts = parsed
        key = (low, high)
        prev = best.get(key)
        if prev is None or ts > prev[0]:
            best[key] = (ts, f)
    return [f for (_, f) in best.values()]


def gather_files(
    directory: Path,
    files: list[str] | None,
    includes: list[str] | None,
    excludes: list[str] | None,
    keep_latest: bool,
    pattern: str = "peaks_*.csv",
) -> list[Path]:
    candidates: set[Path] = set()
    if files:
        for pat in files:
            candidates.update(directory.glob(pat))
    else:
        candidates.update(directory.glob(pattern))

    if includes:
        filtered = set()
        for f in candidates:
            if any(fnmatch.fnmatch(f.name, pat) for pat in includes):
                filtered.add(f)
        candidates = filtered

    if excludes:
        candidates = {
            f for f in candidates if not any(fnmatch.fnmatch(f.name, pat) for pat in excludes)
        }

    files_list = sorted(candidates)
    if keep_latest:
        files_list = latest_per_bin(files_list)

    return files_list


def _run_one_file(
    file_path: Path,
    args,
    ts: str,
    *,
    out_dir: Path | None = None,
    out_path_override: Path | None = None,
) -> pd.DataFrame:
    if out_path_override is not None:
        out_csv_path = out_path_override
    else:
        target_dir = out_dir if out_dir is not None else file_path.parent
        out_csv_path = target_dir / f"{file_path.stem}_filtered_{ts}.csv"

    df = filter_csv(
        csv_path=file_path,
        out_csv_path=out_csv_path,
        band=args.band,
        peak_mode="biweight" if args.biweight else "naive",
        asassn_csv=args.asassn_csv,
        vsx_csv=args.vsx_csv,
        min_dip_fraction=args.min_dip_fraction,
        min_cameras=args.min_cameras,
        max_power=args.max_power,
        min_period=args.min_period,
        max_period=args.max_period,
        min_time_span=200.0,
        min_points_per_day=0.05,
        min_sigma=3.0,
        match_radius_arcsec=args.match_radius_arcsec,
        apply_bns=args.apply_bns,
        apply_vsx_class=args.apply_vsx_class,
        apply_dip_dom=args.apply_dip_dom,
        apply_multi_camera=args.apply_multi_camera,
        apply_periodic=args.apply_periodic,
        apply_sparse=args.apply_sparse,
        apply_sigma=args.apply_sigma,
        seed_workers=args.seed_workers,
        tqdm_position_base=0,
    )

    print(f"[OK] {file_path} → {out_csv_path} ({len(df)} rows)")
    return df


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run filter_csv on one peaks CSV or a directory of peaks_*.csv files.")
    parser.add_argument("csv_path", type=Path, help="Input peaks CSV path OR a directory containing peaks_*.csv files.")
    parser.add_argument("--files", nargs="+", default=None, help="Specific file globs within the directory (e.g., 'peaks_12_5_13_*.csv').")
    parser.add_argument("--include", action="append", default=None, help="Additional basename globs to include; can be given multiple times.")
    parser.add_argument("--exclude", action="append", default=None, help="Basename globs to exclude; can be given multiple times.")
    parser.add_argument("--latest-per-bin", action="store_true", help="If multiple files exist per mag bin, keep only the latest timestamp per bin.")
    parser.add_argument("--dry-run", action="store_true", help="List the files that would be processed and exit.")
    parser.add_argument("--output", type=Path, default=None, help="Destination CSV path when processing a single file.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to place output CSVs (per-file and combined).")
    parser.add_argument("--no-combined", action="store_true", help="Do not write a combined CSV when processing a directory.")
    parser.add_argument("--biweight", action="store_true", help="Use biweight peak CSVs (peaks_biweight_*) and biweight filtering.")
    parser.add_argument("--band", default="either", choices=["g", "v", "both", "either"])
    parser.add_argument("--seed-workers", type=int, default=1, help="Worker processes for the initial candidates_with_peaks_naive stage.")
    parser.add_argument("--asassn-csv", default="results_crossmatch/asassn_index_masked_concat_cleaned_20250926_1557.csv")
    parser.add_argument("--vsx-csv", default="results_crossmatch/vsx_cleaned_20250926_1557.csv")
    parser.add_argument("--min-dip-fraction", type=float, default=0.66)
    parser.add_argument("--min-cameras", type=int, default=2)
    parser.add_argument("--max-power", type=float, default=0.5)
    parser.add_argument("--min-period", type=float, default=None)
    parser.add_argument("--max-period", type=float, default=None)
    parser.add_argument("--match-radius-arcsec", type=float, default=3.0)
    parser.add_argument("--no-dip-dom", dest="apply_dip_dom", action="store_false", default=False, help="Disable the dip-dominated fraction filter.")
    parser.add_argument("--no-multi-camera", dest="apply_multi_camera", action="store_false", default=True, help="Disable the multi-camera filter.")
    parser.add_argument("--no-bns", dest="apply_bns", action="store_false", default=True, help="Disable the bright-nearby-star catalog join.")
    parser.add_argument("--no-vsx-class", dest="apply_vsx_class", action="store_false", default=True, help="Disable VSX variability class enrichment.")
    parser.add_argument("--no-periodic", dest="apply_periodic", action="store_false", default=False, help="Disable the periodicity filter.")
    parser.add_argument("--no-sparse", dest="apply_sparse", action="store_false", default=False, help="Disable the sparse light-curve filter.")
    parser.add_argument("--no-sigma", dest="apply_sigma", action="store_false", default=False, help="Disable the sigma-based depth filter.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    ts = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S%z")

    in_path = args.csv_path
    output_override = args.output.expanduser() if args.output is not None else None
    output_dir = args.output_dir.expanduser() if args.output_dir is not None else None

    if output_override and in_path.is_dir():
        parser.error("--output is only valid when processing a single input file.")
    if output_dir and output_dir.exists() and not output_dir.is_dir():
        parser.error("--output-dir must be a directory.")

    if in_path.is_dir():
        files = gather_files(
            directory=in_path,
            files=args.files,
            includes=args.include,
            excludes=args.exclude,
            keep_latest=args.latest_per_bin,
            pattern="peaks_biweight_*.csv" if args.biweight else "peaks_*.csv",
        )
        if not files:
            parser.error(f"No matching files in {in_path}")

        if args.dry_run:
            print("Files that would be processed:")
            for f in files:
                print("  ", f)
            return 0

        dfs: list[pd.DataFrame] = []
        for f in files:
            dfs.append(_run_one_file(f, args, ts, out_dir=output_dir))

        if not args.no_combined:
            combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            combined_dir = output_dir if output_dir is not None else in_path
            combined_csv = combined_dir / f"peaks_all_filtered_{ts}.csv"
            combined_dir.mkdir(parents=True, exist_ok=True)
            combined.to_csv(combined_csv, index=False)
            print(f"[COMBINED] {combined_csv} ({len(combined)} rows)")
        return 0

    if in_path.is_file() or not in_path.exists():
        df = _run_one_file(
            in_path,
            args,
            ts,
            out_dir=output_dir,
            out_path_override=output_override,
        )
        print(f"Filtered rows: {len(df)}")
        return 0

    parser.error(f"{in_path} is neither a file nor a directory")


if __name__ == "__main__":
    raise SystemExit(main())
