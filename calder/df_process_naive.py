from pathlib import Path
import pandas as pd
import numpy as np

from astropy.timeseries import LombScargle

from lc_utils import read_lc_dat, read_lc_raw
from vsx_crossmatch import propagate_asassn_coords, vsx_coords

def candidates_with_peaks_naive(csv_path, out_csv_path=None, write_csv: bool = True, index: bool = False, band: str = "either",):
    """
    Read peaks_[mag_bin].csv and return only rows where either band has a non-zero number of peaks. Optionally, output peaks_[mag_bin]_selected_dippers.csv. Optionally search for only g band, only v band, or both.

    csv_path can optionally point to a file. if file w/o timestamp is passed, uses most recent timestamp
    """
    file = Path(csv_path)
    if not file.exists():
        suffix = file.suffix or ".csv"
        stem = file.stem
        pattern = f"{stem}_*{suffix}"
        candidates = sorted(file.parent.glob(pattern))
        if not candidates:
            raise FileNotFoundError(f"No file found matching {file} or {pattern}")
        file = max(candidates, key=lambda p: p.stat().st_mtime)

    df = pd.read_csv(file).copy()

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

    return out

def filter_bns(
    df: pd.DataFrame,
    asassn_csv: str | Path = "results_crossmatch/asassn_index_masked_concat_cleaned_20250926_1557.csv",
):
    """
    Keep only rows with matching ASAS-SN catalog entries and append key columns
    """
    catalog = pd.read_csv(asassn_csv)
    catalog["asas_sn_id"] = catalog["asas_sn_id"].astype(str)

    cols_to_attach = [
        "ra_deg",
        "dec_deg",
        "pm_ra",
        "pm_ra_d",
        "pm_dec",
        "pm_dec_d",
    ]
    cols_available = ["asas_sn_id"] + [c for c in cols_to_attach if c in catalog.columns]

    df_out = (
        df.assign(asas_sn_id=df["asas_sn_id"].astype(str))
        .merge(catalog[cols_available], on="asas_sn_id", how="inner")
    )
    df_out = df_out.dropna(subset=[c for c in ["pm_ra", "pm_dec"] if c in df_out.columns])
    return df_out.reset_index(drop=True)

def vsx_class_extract(
    df: pd.DataFrame,
    vsx_csv: str | Path = "results_crossmatch/vsx_cleaned_20250926_1557.csv",
    match_radius_arcsec: float = 3.0,
):
    """
    Append VSX classes for matches within the given radius
    """
    vsx = pd.read_csv(vsx_csv)
    vsx = vsx.dropna(subset=["ra", "dec"]).reset_index(drop=True)

    coords_asassn = propagate_asassn_coords(df)
    coords_vsx = vsx_coords(vsx)

    idx, sep2d, _ = coords_asassn.match_to_catalog_sky(coords_vsx)
    mask = sep2d < (match_radius_arcsec * u.arcsec)

    df_out = df.copy()
    df_out["vsx_match_sep_arcsec"] = sep2d.arcsec
    df_out["vsx_class"] = pd.Series(index=df_out.index, dtype=object)
    if "class" in vsx.columns:
        df_out.loc[mask, "vsx_class"] = vsx.loc[idx[mask], "class"].values
    return df_out


def filter_dip_dominated(df, min_dip_fraction=0.66):
    """
    only keep rows that are dip_dominated by min_dip_fraction in at least one band
    """
    def row_is_dip_dominated(row):
        for col in ("g_dip_fraction", "v_dip_fraction"):
            if col in row and pd.notna(row[col]) and row[col] >= min_dip_fraction:
                return True
        return False

    mask = df.apply(row_is_dip_dominated, axis=1)
    return df.loc[mask].reset_index(drop=True)



def filter_multi_camera(df, min_cameras=2):
    """ 
    require each candidate to be observed by at least min_cameras distinct cameras using .raw summary
    """
    counts = [
        len(read_lc_raw(row["asas_sn_id"], str(Path(row["raw_path"]).parent))["camera#"].unique())
        for _, row in df.iterrows()
    ]
    df = df.copy()
    df["n_cameras"] = counts
    return df.loc[df["n_cameras"] >= min_cameras].reset_index(drop=True)


def filter_periodic_candidates(df, max_power=0.5, min_period=None, max_period=None):
    """
    reject candidates whose lsp shows max_power periodicity; defaults to g band, then falls back to v band
    """
    keep = []
    for idx, row in df.iterrows():
        dfg, dfv = read_lc_dat(row["asas_sn_id"], row["lc_dir"])
        work = dfg if not dfg.empty else dfv
        if work.empty:
            continue
        times = work["JD"].to_numpy(dtype=float)
        mags = work["mag"].to_numpy(dtype=float)
        mags -= np.nanmean(mags)
        ls = LombScargle(times, mags)
        freq, power = ls.autopower()
        if min_period is not None and max_period is not None and power.size > 0:
            period = 1.0 / freq
            valid = (period >= min_period) & (period <= max_period)
            power = power[valid] if valid.any() else power
        if power.size == 0 or np.nanmax(power) <= max_power:
            keep.append(idx)
    return df.loc[keep].reset_index(drop=True)


def filter_sparse_lightcurves(df, min_time_span=200.0, min_points_per_day=0.05):
    """
    remove lc's with too little coverage or sparse cadence
    """
    spans = df["jd_last"] - df["jd_first"]
    mask = spans >= min_time_span
    spans = spans.where(spans > 0, np.nan)

    for band in ("g", "v"):
        n_rows_col = f"n_rows_{band}"
        if n_rows_col in df.columns:
            points_per_day = df[n_rows_col] / spans
            mask &= points_per_day >= min_points_per_day

    return df.loc[mask].reset_index(drop=True)


# double check this. there's already a robust scatter computed so using the raw summary widths may be dumb
def filter_sigma_resid(df, min_sigma=3.0):
    """
    keep rows w/ strongest dip exceeding min_sigma wrt the per-camera scatter; approximates sigma using raw summary widths
    """
    keep = []
    for idx, row in df.iterrows():
        raw_df = read_lc_raw(row["asas_sn_id"], str(Path(row["raw_path"]).parent))
        scatter_vals = (raw_df["sig1_high"] - raw_df["sig1_low"]).to_numpy(dtype=float)
        finite = scatter_vals[np.isfinite(scatter_vals)]
        if finite.size == 0:
            continue
        scatter = np.nanmedian(finite)
        for depth_col in ("g_max_depth", "v_max_depth"):
            if depth_col in df.columns and pd.notna(row[depth_col]):
                if (row[depth_col] / scatter) >= min_sigma:
                    keep.append(idx)
                    break
    return df.loc[keep].reset_index(drop=True)


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
    skip_bns: bool = False,
    skip_vsx_class: bool = False,
    skip_dip_dom: bool = False,
    skip_multi_camera: bool = False,
    skip_periodic: bool = False,
    skip_sparse: bool = False,
    skip_sigma: bool = False,
) -> pd.DataFrame:
    
    df_filtered = candidates_with_peaks_naive(csv_path, write_csv=False, band=band)
    if not skip_bns:
        df_filtered = filter_bns(df_filtered, asassn_csv=asassn_csv)
    if not skip_vsx_class:
        df_filtered = vsx_class_extract(
            df_filtered,
            vsx_csv=vsx_csv,
            match_radius_arcsec=match_radius_arcsec,
        )
    if not skip_dip_dom:
        df_filtered = filter_dip_dominated(
            df_filtered, min_dip_fraction=min_dip_fraction
        )
    if not skip_multi_camera:
        df_filtered = filter_multi_camera(df_filtered, min_cameras=min_cameras)
    if not skip_periodic:
        df_filtered = filter_periodic_candidates(
            df_filtered,
            max_power=max_power,
            min_period=min_period,
            max_period=max_period,
        )
    if not skip_sparse:
        df_filtered = filter_sparse_lightcurves(
            df_filtered,
            min_time_span=min_time_span,
            min_points_per_day=min_points_per_day,
        )
    if not skip_sigma:
        df_filtered = filter_sigma_resid(df_filtered, min_sigma=min_sigma)
    df_filtered = df_filtered.reset_index(drop=True)

    if out_csv_path is not None:
        Path(out_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df_filtered.to_csv(out_csv_path, index=False)
    
    return df_filtered
