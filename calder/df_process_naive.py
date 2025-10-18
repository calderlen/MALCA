from pathlib import Path
import pandas as pd
import numpy as np

from astropy.timeseries import LombScargle

from lc_utils import read_lc_dat, read_raw_summary


# filtering steps to add
    
    # check ra, dec of candidates against vsx crossmatch, see if they match up with any vsx variable types, then optionally, later, filter based on that. do this only after all of the other filtering steps


def candidates_with_peaks_naive(
    csv_path,
    out_csv_path=None,
    write_csv: bool = True,
    index: bool = False,
    band: str = "either",
):
    """
    Read peaks_[mag_bin].csv and return only rows where either band has a non-zero number of peaks. Optionally, output peaks_[mag_bin]_selected_dippers.csv. Optionally search for only g band, only v band, or both.

    csv_path can optionally point to a file. if file w/o timestamp is passed, uses most recent timestamp
    """

    file = Path(csv_path)

    if not file.exists():
        # if file w/o timestamp is passed, uses most recent timestamp
        suffix = file.suffix or ".csv"
        stem = file.stem
        pattern = f"{stem}_*{suffix}"
        candidates = sorted(file.parent.glob(pattern))
        if not candidates:
            raise FileNotFoundError(
                f"No file found matching {file} or {pattern}"
            )
        file = max(candidates, key=lambda p: p.stat().st_mtime)

    df = pd.read_csv(file).copy()

    for col in ("g_n_peaks", "v_n_peaks"):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' is missing; cannot select nonzero-peak rows."
            )

    df["g_n_peaks"] = pd.to_numeric(df["g_n_peaks"], errors="coerce").fillna(0)
    df["v_n_peaks"] = pd.to_numeric(df["v_n_peaks"], errors="coerce").fillna(0)

    band_key = band.lower()
    if band_key not in {"g", "v", "both", "either"}:
        raise ValueError("band must be one of 'g', 'v', 'both', 'either'")

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

def filter_dip_dominated(df, min_dip_fraction=0.66, bands=("g", "v")):
    """
    only keep rows that are dip_dominated by min_dip_fraction in at least one band
    """
    def _row_is_dip_dominated(row):
        for band in bands:
            col = f"{band}_dip_fraction"
            if col in row and pd.notna(row[col]) and row[col] >= min_dip_fraction:
                return True
        return False

    mask = df.apply(_row_is_dip_dominated, axis=1)
    return df.loc[mask].reset_index(drop=True)



def filter_multi_camera(df, min_cameras=2):
    """ 
    require each candidate to be observed by at least min_cameras distinct cameras using .raw summary
    """
    counts = []
    for _, row in df.iterrows():
        raw_path = row.get("raw_path")
        if pd.isna(raw_path):
            counts.append(np.nan)
            continue
        raw_df = read_raw_summary(row["asas_sn_id"], str(Path(raw_path).parent))
        counts.append(len(raw_df["camera#"].unique()) if not raw_df.empty else np.nan)

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
            keep.append(idx)
            continue
        times = work["JD"].to_numpy(dtype=float)
        mags = work["mag"].to_numpy(dtype=float)
        mags -= np.nanmean(mags)
        try:
            ls = LombScargle(times, mags)
            freq, power = ls.autopower()
        except Exception:
            keep.append(idx)
            continue
        if min_period is not None and max_period is not None and power.size > 0:
            period = 1.0 / freq
            valid = (period >= min_period) & (period <= max_period)
            power = power[valid] if valid.any() else power
        if power.size == 0 or np.nanmax(power) <= max_power:
            keep.append(idx)
    return df.loc[keep].reset_index(drop=True)


def filter_sparse_lightcurves(
    df, min_time_span=200.0, min_points_per_day=0.05, bands=("g", "v")
):
    """
    remove lc's with too little coverage or sparse cadence
    """
    spans = df["jd_last"] - df["jd_first"]
    mask = spans >= min_time_span
    spans = spans.where(spans > 0, np.nan)

    for band in bands:
        n_rows_col = f"n_rows_{band}"
        if n_rows_col not in df.columns:
            continue
        points_per_day = df[n_rows_col] / spans
        mask &= points_per_day >= min_points_per_day

    return df.loc[mask].reset_index(drop=True)



# double check this. there's already a robust scatter computed so using the raw summary widths may be dumb
def filter_sigma_resid(df, min_sigma=3.0, bands=("g", "v")):
    """
    keep rows w/ strongest dip exceeding min_sigma wrt the per-camera scatter; approximates sigma using raw summary widths
    """
    keep = []
    for idx, row in df.iterrows():
        raw_path = row.get("raw_path")
        if pd.isna(raw_path):
            continue
        raw_df = read_raw_summary(row["asas_sn_id"], str(Path(raw_path).parent))
        if raw_df.empty:
            continue
        scatter_vals = (raw_df["sig1_high"] - raw_df["sig1_low"]).to_numpy(dtype=float)
        scatter = np.nanmedian(scatter_vals[np.isfinite(scatter_vals)])
        if scatter <= 0 or np.isnan(scatter):
            continue
        for band in bands:
            depth_col = f"{band}_max_depth"
            if depth_col in df.columns and pd.notna(row[depth_col]):
                if (row[depth_col] / scatter) >= min_sigma:
                    keep.append(idx)
                    break
    return df.loc[keep].reset_index(drop=True)

def filter_df(
    df,
    *,
    min_rows_g=None,
    min_rows_v=None,
    min_g_n_peaks=None,
    min_v_n_peaks=None,
    min_g_dip_fraction=None,
    max_g_jump_fraction=None,
    min_v_dip_fraction=None,
    max_v_jump_fraction=None,
    min_g_n_dip_runs=None,
    min_v_n_dip_runs=None,
    min_g_max_depth=None,
    min_v_max_depth=None,
    require_g_dip_dominated=None,
    require_v_dip_dominated=None,
):
    """
    general filtration function, intakes intake of raw list of sources and outputs fitlered df that nominally contains candidates only

    require_[band]_dip_dominated is a boolean
    """

    mask = pd.Series(True, index=df.index)

    def _col(name):
        if name not in df.columns:
            raise KeyError(
                f"Column '{name}' is missing; regenerate parquet with dip_finder to include it."
            )
        return df[name]

    def _gte(name, value):
        nonlocal mask
        if value is None:
            return
        mask &= _col(name).fillna(-np.inf) >= value

    def _lte(name, value):
        nonlocal mask
        if value is None:
            return
        mask &= _col(name).fillna(np.inf) <= value

    def _require_bool(name, desired):
        nonlocal mask
        if desired is None:
            return
        series = _col(name).fillna(False).astype(bool)
        mask &= (series == bool(desired))

    # point-count constraints
    _gte("n_rows_g", min_rows_g)
    _gte("n_rows_v", min_rows_v)

    # peak counts
    _gte("g_n_peaks", min_g_n_peaks)
    _gte("v_n_peaks", min_v_n_peaks)

    # dip/jump fractions
    _gte("g_dip_fraction", min_g_dip_fraction)
    _lte("g_jump_fraction", max_g_jump_fraction)
    _gte("v_dip_fraction", min_v_dip_fraction)
    _lte("v_jump_fraction", max_v_jump_fraction)

    # run counts
    _gte("g_n_dip_runs", min_g_n_dip_runs)
    _gte("v_n_dip_runs", min_v_n_dip_runs)

    # depth thresholds
    _gte("g_max_depth", min_g_max_depth)
    _gte("v_max_depth", min_v_max_depth)

    # dip-dominated requirement
    _require_bool("g_is_dip_dominated", require_g_dip_dominated)
    _require_bool("v_is_dip_dominated", require_v_dip_dominated)

    return df.loc[mask].reset_index(drop=True)
