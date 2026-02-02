import os
import re
from glob import glob

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import mad_std, sigma_clip
from tqdm import tqdm

colors = ["#6b8bcd", "#b3b540", "#8f62ca", "#5eb550", "#c75d9c", "#4bb092", "#c5562f", "#6c7f39",
              "#ce5761", "#c68c45", '#b5b246', '#d77fcc', '#7362cf', '#ce443f', '#3fc1bf', '#cda735',
              '#a1b055']


def gaussian(t, amp, t0, sigma, baseline):
    """
    Gaussian kernel + baseline term (shared between events and dipper scoring).
    """
    return baseline + amp * np.exp(-0.5 * ((t - t0) / sigma) ** 2)


def paczynski_kernel(t, amp, t0, tE, baseline):
    """
    Simple Paczyński kernel approximation + baseline term.

    Fast approximation for curve fitting. For full physical microlensing
    model with magnification, see malca.score.paczynski().

    Parameters
    ----------
    t : array-like
        Time values
    amp : float
        Amplitude
    t0 : float
        Time of peak
    tE : float
        Einstein crossing time (characteristic timescale)
    baseline : float
        Baseline magnitude

    Returns
    -------
    array-like
        Model magnitudes
    """
    tE = np.maximum(np.abs(tE), 1e-5)
    return baseline + amp / np.sqrt(1.0 + ((t - t0) / tE) ** 2)


def fred(t, amp, t0, tau, baseline):
    """
    Fast Rise Exponential Decay (FRED) kernel + baseline term.

    Parameters
    ----------
    t : array-like
        Time values
    amp : float
        Amplitude
    t0 : float
        Time of peak (start of decay)
    tau : float
        Decay timescale
    baseline : float
        Baseline magnitude

    Returns
    -------
    array-like
        Model magnitudes
    """
    tau = np.maximum(np.abs(tau), 1e-5)
    dt = t - t0
    # Mask out values before t0
    decay = np.where(dt >= 0, np.exp(-dt / tau), 0.0)
    return baseline + amp * decay


def skew_gaussian(t, amp, t0, sigma, baseline, alpha):
    """
    Skew-normal distribution kernel + baseline term.

    Parameters
    ----------
    t : array-like
        Time values
    amp : float
        Amplitude of the dip (positive for dimming)
    t0 : float
        Center time of the event
    sigma : float
        Width parameter (like standard deviation)
    baseline : float
        Baseline magnitude
    alpha : float
        Skewness parameter. alpha=0 is symmetric Gaussian.
        alpha>0 skews right (slower egress), alpha<0 skews left (slower ingress).

    Returns
    -------
    array-like
        Model magnitudes
    """
    from scipy.special import erf
    sigma = np.maximum(np.abs(sigma), 1e-5)
    z = (t - t0) / sigma
    # Skew-normal: gaussian * (1 + erf(alpha * z / sqrt(2)))
    # Normalized so peak amplitude matches 'amp' approximately
    skew_factor = 1 + erf(alpha * z / np.sqrt(2))
    return baseline + amp * np.exp(-0.5 * z**2) * skew_factor


def get_id_col(df: pd.DataFrame) -> str:
    """Find the ID column in a dataframe."""
    for candidate in ["asas_sn_id", "id", "source_id", "path"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("No ID column found. Expected one of: asas_sn_id, id, source_id, path")


def clean_lc(df, max_error_absolute=1.0, max_error_sigma=5.0):
    base_mask = np.ones(len(df), dtype=bool)
    base_mask &= (df["saturated"] == 0)
    base_mask &= df["JD"].notna() & df["mag"].notna()
    base_mask &= df["error"].notna() & (df["error"] > 0.0)

    mask = base_mask.copy()
    if base_mask.sum() > 0:
        errors = df.loc[base_mask, "error"].values
        mask &= (df["error"] < max_error_absolute)

        clipped = sigma_clip(
            errors,
            sigma=max_error_sigma,
            cenfunc="median",
            stdfunc=mad_std,
        )
        clipped_mask = np.asarray(clipped.mask)
        if clipped_mask.shape == errors.shape:
            # Create a full-length mask for clipped errors
            clipped_full = np.zeros(len(df), dtype=bool)
            clipped_full[base_mask] = clipped_mask
            mask &= ~clipped_full

    df = df.loc[mask]
    df = df.sort_values("JD").reset_index(drop=True)
    return df


def compute_time_stats(df_lc: pd.DataFrame) -> dict:
    """Compute time span and cadence stats from a light curve DataFrame."""
    if df_lc.empty:
        return {"time_span_days": 0.0, "points_per_day": 0.0}

    jd = df_lc["JD"].values
    jd = jd[np.isfinite(jd)]
    if len(jd) < 2:
        return {"time_span_days": 0.0, "points_per_day": 0.0}

    time_span_days = float(jd.max() - jd.min())
    points_per_day = len(jd) / time_span_days if time_span_days > 0 else 0.0

    return {
        "time_span_days": time_span_days,
        "points_per_day": points_per_day,
    }


def compute_n_cameras(df_lc: pd.DataFrame) -> int:
    """Count unique cameras from a light curve DataFrame."""
    if df_lc.empty:
        return 0
    cameras = df_lc["camera#"].dropna().unique()
    return int(len(cameras))


def year_to_jd(year):
    jd_epoch = 2449718.5
    year_epoch = 1995
    days_in_year = 365.25

    return (year - year_epoch) * days_in_year + (jd_epoch - 2450000.0)


def jd_to_year(jd):
    jd_epoch = 2449718.5
    year_epoch = 1995
    days_in_year = 365.25

    return year_epoch + ((jd + 2450000.0) - jd_epoch) / days_in_year


def read_lc_dat2(asassn_id, path, excluded_cameras: set[int] | str | None = None):
    """
    Read light curve data from .dat2 file.

    Parameters
    ----------
    asassn_id : str
        ASAS-SN source ID
    path : str
        Path to directory containing the .dat2 file
    excluded_cameras : set[int] | str | None
        Camera IDs to exclude from the output. Can be:
        - None: no filtering
        - set of ints: camera IDs to exclude
        - comma-separated string: "1,6,7" -> exclude cameras 1, 6, 7

    Returns
    -------
    df_g, df_v : pd.DataFrame
        g-band and V-band DataFrames with excluded cameras removed
    """
    # Parse excluded_cameras if string
    if isinstance(excluded_cameras, str) and excluded_cameras:
        excluded_cameras = {int(c.strip()) for c in excluded_cameras.split(",") if c.strip()}
    elif excluded_cameras is None:
        excluded_cameras = set()

    dat2_path = os.path.join(path, f"{asassn_id}.dat2")
    if os.path.exists(dat2_path):
        file = dat2_path
                      
        columns = ["JD",
                   "mag",
                   "error",
                   "good_bad",
                   "camera#",
                   "v_g_band",
                   "saturated",
                   "cam_field"]

        # Use whitespace delimiter instead of fixed-width to handle
        # variable-width JD values (4-digit vs 5-digit integer parts).
        # read_fwf infers column widths from early rows, which fails when
        # JD transitions from 9999 to 10000+ (leading digit gets truncated).
        df = pd.read_csv(
            file,
            header=None,
            names=columns,
            sep=r'\s+',
        )
    
                                               
        df[["camera_name", "field"]] = df["cam_field"].str.split("/", expand=True)

                                      
        df = df.drop(columns=["cam_field"])

                        
        df = df.astype({
            "JD": "float64",
            "mag": "float64",
            "error": "float64",
            "good_bad": "int64",
            "camera#": "int64",
            "v_g_band": "int64",
            "saturated": "int64",
            "camera_name": "string",
            "field": "string"
        })

        # Filter out excluded cameras
        if excluded_cameras:
            df = df[~df["camera#"].isin(excluded_cameras)]

        df_g = df.loc[df["v_g_band"] == 0].reset_index(drop=True)    
        df_v = df.loc[df["v_g_band"] == 1].reset_index(drop=True)

        return df_g, df_v

    raise FileNotFoundError(
        f"Light curve file not found: {dat2_path}"
    )


def read_lc_csv(asassn_id, path):
    csv_path = os.path.join(path, f"{asassn_id}.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(csv_path)

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df["JD"] = df["jd"] + 2450000.0

    df_g = df[df["phot_filter"] == "g"].copy().reset_index(drop=True)
    df_v = df[df["phot_filter"] == "V"].copy().reset_index(drop=True)

    return df_g, df_v


def read_lc_raw(asassn_id, path):
    raw_path = os.path.join(path, f"{asassn_id}.raw")
    if not os.path.exists(raw_path):
        return pd.DataFrame()
    columns = [
        "camera#",
        "median",
        "sig1_low",
        "sig1_high",
        "p90_low",
        "p90_high",
    ]
    df = pd.read_csv(
        raw_path,
        sep=r"\s+",
        header=None,
        names=columns,
        dtype={
            "camera#": "int64",
            "median": "float64",
            "sig1_low": "float64",
            "sig1_high": "float64",
            "p90_low": "float64",
            "p90_high": "float64",
        },
    )
    return df


def read_lc_raw2(asassn_id, path):
    """
    Read per-camera statistics from .raw2 file.

    The .raw2 file contains per-camera scatter statistics from the original raw data:
    cam# median 1siglow 1sighigh 90percentlow 90percenthigh

    Parameters
    ----------
    asassn_id : str
        ASAS-SN source ID
    path : str
        Path to directory containing the .raw2 file

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: camera#, median, sig1_low, sig1_high, p90_low, p90_high
    """
    raw2_path = os.path.join(path, f"{asassn_id}.raw2")
    if not os.path.exists(raw2_path):
        return pd.DataFrame()
    columns = [
        "camera#",
        "median",
        "sig1_low",
        "sig1_high",
        "p90_low",
        "p90_high",
    ]
    df = pd.read_csv(
        raw2_path,
        sep=r"\s+",
        header=None,
        names=columns,
        dtype={
            "camera#": "int64",
            "median": "float64",
            "sig1_low": "float64",
            "sig1_high": "float64",
            "p90_low": "float64",
            "p90_high": "float64",
        },
    )
    # Compute expected scatter (1-sigma half-width)
    df["expected_scatter"] = (df["sig1_high"] - df["sig1_low"]) / 2.0
    return df


def identify_bad_cameras(
    df_lc: pd.DataFrame,
    raw2_df: pd.DataFrame | None = None,
    *,
    window_days: float = 100.0,
    min_overlap_points: int = 10,
    scatter_ratio_threshold: float = 2.5,
    min_cameras_for_comparison: int = 2,
    cam_col: str = "camera#",
    t_col: str = "JD",
    mag_col: str = "mag",
) -> set[int]:
    """
    Identify cameras with anomalously high scatter compared to other cameras
    in overlapping time windows.

    This avoids filtering out cameras that are the only ones observing during
    a real event - only cameras with high scatter RELATIVE to other cameras
    in the same time window are flagged.

    Algorithm:
    1. For each camera, find all time windows where it overlaps with other cameras
    2. In each overlap window, compute MAD scatter for this camera and others
    3. If this camera's scatter is consistently >threshold times the median of others,
       flag it as bad

    Parameters
    ----------
    df_lc : pd.DataFrame
        Light curve data with JD, mag, camera# columns
    raw2_df : pd.DataFrame | None
        Optional raw2 data with expected scatter per camera
    window_days : float
        Size of sliding window for overlap comparison (days)
    min_overlap_points : int
        Minimum points required in overlap window for comparison
    scatter_ratio_threshold : float
        Threshold for scatter ratio (camera scatter / median other scatter)
    min_cameras_for_comparison : int
        Minimum number of other cameras needed for valid comparison
    cam_col, t_col, mag_col : str
        Column names

    Returns
    -------
    set[int]
        Camera IDs identified as having anomalously high scatter
    """
    if df_lc.empty:
        return set()

    cameras = df_lc[cam_col].dropna().unique()
    if len(cameras) < 2:
        return set()

    # Build per-camera data
    cam_data = {}
    for cam in cameras:
        cam_df = df_lc[df_lc[cam_col] == cam]
        t = cam_df[t_col].values
        m = cam_df[mag_col].values
        finite = np.isfinite(t) & np.isfinite(m)
        cam_data[cam] = {
            "t": t[finite],
            "m": m[finite],
            "t_min": t[finite].min() if finite.any() else np.inf,
            "t_max": t[finite].max() if finite.any() else -np.inf,
        }

    # Track scatter ratios for each camera
    cam_scatter_ratios = {cam: [] for cam in cameras}

    # Sliding window comparison
    t_all = df_lc[t_col].dropna().values
    t_start = t_all.min()
    t_end = t_all.max()
    step = window_days / 2  # 50% overlap

    t_window = t_start
    while t_window < t_end:
        t_lo = t_window
        t_hi = t_window + window_days

        # Find cameras with data in this window
        cams_in_window = []
        for cam, data in cam_data.items():
            t_cam = data["t"]
            in_window = (t_cam >= t_lo) & (t_cam <= t_hi)
            n_in_window = in_window.sum()
            if n_in_window >= min_overlap_points:
                # Compute scatter (MAD) in this window
                m_window = data["m"][in_window]
                med = np.median(m_window)
                scatter = 1.4826 * np.median(np.abs(m_window - med))
                cams_in_window.append((cam, scatter, n_in_window))

        # Compare each camera to others in this window
        if len(cams_in_window) >= min_cameras_for_comparison + 1:
            scatters = np.array([s for _, s, _ in cams_in_window])
            for i, (cam, scatter, _) in enumerate(cams_in_window):
                # Compute median scatter of OTHER cameras
                other_scatters = np.concatenate([scatters[:i], scatters[i+1:]])
                if len(other_scatters) >= min_cameras_for_comparison:
                    median_other = np.median(other_scatters)
                    if median_other > 0:
                        ratio = scatter / median_other
                        cam_scatter_ratios[cam].append(ratio)

        t_window += step

    # Identify bad cameras: those with consistently high scatter ratios
    bad_cameras = set()
    for cam, ratios in cam_scatter_ratios.items():
        if len(ratios) >= 3:  # Need at least 3 comparisons
            # Use median ratio to be robust to outliers
            median_ratio = np.median(ratios)
            if median_ratio > scatter_ratio_threshold:
                bad_cameras.add(int(cam))

    # Optional: also check against raw2 expected scatter
    if raw2_df is not None and not raw2_df.empty:
        for cam in cameras:
            if cam in bad_cameras:
                continue
            cam_df = df_lc[df_lc[cam_col] == cam]
            m = cam_df[mag_col].dropna().values
            if len(m) < 10:
                continue
            actual_scatter = 1.4826 * np.median(np.abs(m - np.median(m)))

            # Get expected scatter from raw2
            raw2_row = raw2_df[raw2_df["camera#"] == cam]
            if raw2_row.empty:
                continue
            expected = raw2_row["expected_scatter"].values[0]
            if expected > 0 and actual_scatter > 3 * expected:
                # Actual scatter is way higher than expected from raw data
                bad_cameras.add(int(cam))

    return bad_cameras


def identify_offset_cameras(
    df_lc: pd.DataFrame,
    *,
    window_days: float = 100.0,
    min_overlap_points: int = 10,
    offset_sigma_threshold: float = 15.0,
    min_cameras_for_comparison: int = 2,
    cam_col: str = "camera#",
    t_col: str = "JD",
    mag_col: str = "mag",
    remove_full_camera: bool = True,
) -> tuple[set[int], set[tuple[int, float, float]]]:
    """
    Identify cameras with systematic median offsets from other cameras.

    Compares each camera's median magnitude within rolling windows to the
    consensus median of all other cameras. If a camera's median is >N sigma
    discrepant from the consensus, it is flagged.

    Parameters
    ----------
    df_lc : pd.DataFrame
        Light curve data with JD, mag, camera# columns
    window_days : float
        Size of sliding window for overlap comparison (days)
    min_overlap_points : int
        Minimum points required in overlap window for comparison
    offset_sigma_threshold : float
        Flag camera if its median is this many sigma from consensus
    min_cameras_for_comparison : int
        Minimum number of other cameras needed for valid comparison
    cam_col, t_col, mag_col : str
        Column names
    remove_full_camera : bool
        If True, return camera IDs to remove entirely. If False, also return
        specific (camera, window_start, window_end) tuples for targeted removal.

    Returns
    -------
    bad_cameras : set[int]
        Camera IDs flagged for removal (entire camera)
    bad_windows : set[tuple[int, float, float]]
        (camera_id, window_start_jd, window_end_jd) tuples for targeted removal
        Only populated if remove_full_camera=False
    """
    if df_lc.empty or cam_col not in df_lc.columns:
        return set(), set()

    cameras = df_lc[cam_col].dropna().unique()
    if len(cameras) < min_cameras_for_comparison + 1:
        return set(), set()

    jd_min = df_lc[t_col].min()
    jd_max = df_lc[t_col].max()

    # Track offset violations per camera
    cam_offset_violations: dict[int, list[tuple[float, float]]] = {int(c): [] for c in cameras}

    # Slide window across time range
    window_start = jd_min
    window_step = window_days / 2  # 50% overlap

    while window_start < jd_max:
        window_end = window_start + window_days
        window_mask = (df_lc[t_col] >= window_start) & (df_lc[t_col] < window_end)
        window_df = df_lc[window_mask]

        if window_df.empty:
            window_start += window_step
            continue

        # Compute median per camera in this window
        cam_medians = {}
        for cam in cameras:
            cam_df = window_df[window_df[cam_col] == cam]
            if len(cam_df) >= min_overlap_points:
                cam_medians[int(cam)] = np.median(cam_df[mag_col].dropna().values)

        if len(cam_medians) < min_cameras_for_comparison + 1:
            window_start += window_step
            continue

        # For each camera, compare to consensus of others
        for cam, median_val in cam_medians.items():
            other_medians = [v for k, v in cam_medians.items() if k != cam]
            if len(other_medians) < min_cameras_for_comparison:
                continue

            consensus_median = np.median(other_medians)
            consensus_mad = 1.4826 * np.median(np.abs(np.array(other_medians) - consensus_median))

            if consensus_mad < 0.001:  # Avoid division by tiny numbers
                consensus_mad = 0.01

            offset_sigma = abs(median_val - consensus_median) / consensus_mad

            if offset_sigma > offset_sigma_threshold:
                cam_offset_violations[cam].append((window_start, window_end))

        window_start += window_step

    # Determine which cameras to flag
    bad_cameras: set[int] = set()
    bad_windows: set[tuple[int, float, float]] = set()

    for cam, violations in cam_offset_violations.items():
        if len(violations) >= 2:  # Need at least 2 violations to be confident
            if remove_full_camera:
                bad_cameras.add(cam)
            else:
                for start, end in violations:
                    bad_windows.add((cam, start, end))

    return bad_cameras, bad_windows


def filter_bad_cameras(
    df_lc: pd.DataFrame,
    raw2_df: pd.DataFrame | None = None,
    lc_path: str | None = None,
    *,
    filter_scatter: bool = True,
    filter_offset: bool = True,
    offset_sigma_threshold: float = 15.0,
    remove_full_camera: bool = True,
    **kwargs
) -> tuple[pd.DataFrame, set[int]]:
    """
    Filter out cameras with anomalously high scatter or systematic offsets.

    Combines two filters:
    1. Scatter filter (identify_bad_cameras): flags cameras with high MAD scatter
    2. Offset filter (identify_offset_cameras): flags cameras with systematic median offsets

    Parameters
    ----------
    df_lc : pd.DataFrame
        Light curve data
    raw2_df : pd.DataFrame | None
        Optional raw2 data (if provided, takes precedence over lc_path)
    lc_path : str | None
        Path to the light curve file (e.g., .dat2). If provided and raw2_df is
        None, will attempt to load the corresponding .raw2 file automatically.
    filter_scatter : bool
        Apply scatter-based filtering (default: True)
    filter_offset : bool
        Apply median-offset filtering (default: True)
    offset_sigma_threshold : float
        Sigma threshold for offset filtering (default: 5.0)
    remove_full_camera : bool
        If True, remove entire camera when flagged. If False, for offset filter
        only remove the specific offending time windows (default: True)
    **kwargs
        Additional arguments passed to identify_bad_cameras (e.g., scatter_ratio_threshold)

    Returns
    -------
    df_filtered : pd.DataFrame
        DataFrame with bad cameras/points removed
    bad_cameras : set[int]
        Set of camera IDs that were fully removed
    """
    # Auto-load raw2 if lc_path provided and raw2_df not explicitly given
    if raw2_df is None and lc_path is not None:
        from pathlib import Path
        lc_path_obj = Path(lc_path)
        if lc_path_obj.suffix.lower() == ".dat2":
            raw2_path = lc_path_obj.with_suffix(".raw2")
            if raw2_path.exists():
                raw2_df = read_lc_raw2(lc_path_obj.stem, str(lc_path_obj.parent))
    
    cam_col = kwargs.get("cam_col", "camera#")
    t_col = kwargs.get("t_col", "JD")
    bad_cameras: set[int] = set()
    df_filtered = df_lc
    
    # 1. Scatter filter
    if filter_scatter:
        scatter_bad = identify_bad_cameras(df_lc, raw2_df, **kwargs)
        bad_cameras.update(scatter_bad)
    
    # 2. Offset filter
    if filter_offset:
        offset_bad, offset_windows = identify_offset_cameras(
            df_lc if df_filtered is df_lc else df_filtered,
            offset_sigma_threshold=offset_sigma_threshold,
            remove_full_camera=remove_full_camera,
            cam_col=cam_col,
            t_col=t_col,
        )
        bad_cameras.update(offset_bad)
        
        # If targeted removal mode, remove just the offending windows
        if not remove_full_camera and offset_windows:
            mask = pd.Series(True, index=df_filtered.index)
            for cam, start, end in offset_windows:
                # Mark points in this camera within this window for removal
                window_mask = (
                    (df_filtered[cam_col] == cam) &
                    (df_filtered[t_col] >= start) &
                    (df_filtered[t_col] < end)
                )
                mask &= ~window_mask
            df_filtered = df_filtered[mask].reset_index(drop=True)
    
    # Remove full cameras
    if bad_cameras and cam_col in df_filtered.columns:
        df_filtered = df_filtered[~df_filtered[cam_col].isin(bad_cameras)].reset_index(drop=True)
    
    return df_filtered, bad_cameras


def match_index_to_lc(
    index_path: str = "/data/poohbah/1/assassin/lenhart/code/calder/lcsv2_masked/",
    lc_path:    str = "/data/poohbah/1/assassin/rowan.90/lcsv2",
    mag_bins:   list = ['12_12.5','12.5_13','13_13.5','13.5_14','14_14.5','14.5_15'],
    id_column:  str = "asas_sn_id",
):
    """
    Generator function that iterates over index*_masked.csv files in lcsv2_masked/<mag_bin>/, find corresponding lc<num>_cal/ directories in lcsv2/<mag_bin>/, and yield one record per asas_sn_id with whether its .dat file exists. Outputs a dict
    """

    idx_pattern = re.compile(r"index(\d+)_masked\.csv$", re.IGNORECASE)

    for mag_bin in tqdm(mag_bins, desc="Bins", unit="bin"):
        idx_paths = sorted(glob(os.path.join(index_path, mag_bin, "index*_masked.csv")))
        for idx_csv in tqdm(idx_paths, desc=f"{mag_bin} index CSVs", leave=False):

                                                        
            idx_num = int(idx_pattern.search(os.path.basename(idx_csv)).group(1))

            lc_dir = os.path.join(lc_path, mag_bin, f"lc{idx_num}_cal")

            ids = (
                pd.read_csv(idx_csv, dtype={id_column: "string"})[id_column]
                .dropna()
                .astype(str)
                .unique()
            )

            for asn in ids:
                dat_path = os.path.join(lc_dir, f"{asn}.dat")
                found = os.path.exists(dat_path)
                yield {
                    "mag_bin":      mag_bin,
                    "index_num":    idx_num,
                    "index_csv":    idx_csv,
                    "lc_dir":       lc_dir,
                    "asas_sn_id":   asn,
                    "dat_path":     dat_path if found else None,
                    "found":        found,
                }


def custom_id(ra_val,dec_val):
    """
    ADOPTED FROM BRAYDEN JOHANTGEN'S CODE: https://github.com/johantgen13/Dippers_Project.git
    """
    c = SkyCoord(ra=ra_val*u.degree, dec=dec_val*u.degree, frame='icrs')
    ra_num = c.ra.hms
    dec_num = c.dec.dms

    if int(dec_num[0]) < 0:
        cust_id = 'J'+str(int(c.ra.hms[0])).rjust(2,'0')+str(int(c.ra.hms[1])).rjust(2,'0')+str(int(round(c.ra.hms[2]))).rjust(2,'0')+'$-$'+str(int(c.dec.dms[0])*(-1)).rjust(2,'0')+str(int(c.dec.dms[1])*(-1)).rjust(2,'0')+str(int(round(c.dec.dms[2])*(-1))).rjust(2,'0')
    else:
        cust_id = 'J'+str(int(c.ra.hms[0])).rjust(2,'0')+str(int(c.ra.hms[1])).rjust(2,'0')+str(int(round(c.ra.hms[2]))).rjust(2,'0')+'$+$'+str(int(c.dec.dms[0])).rjust(2,'0')+str(int(c.dec.dms[1])).rjust(2,'0')+str(int(round(c.dec.dms[2]))).rjust(2,'0')

    return cust_id


def plotparams(ax, labelsize=15):
    """
    ADAPTED FROM BRAYDEN JOHANTGEN'S CODE: https://github.com/johantgen13/Dippers_Project.git
    """

    ax.minorticks_on()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(direction='in', which='both', labelsize=labelsize)
    ax.tick_params('both', length=8, width=1.8, which='major')
    ax.tick_params('both', length=4, width=1, which='minor')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)
    return ax



def divide_cameras():
    """
    ADAPTED FROM BRAYDEN JOHANTGEN'S CODE: https://github.com/johantgen13/Dippers_Project.git
    """
    pass
