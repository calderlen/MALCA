import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import mad_std, sigma_clip
from tqdm import tqdm

from malca.config.config_filters import (
    CLEAN_LC_MAX_ERROR_ABSOLUTE, CLEAN_LC_MAX_ERROR_SIGMA,
    BAD_CAMERA_WINDOW_DAYS, BAD_CAMERA_MIN_OVERLAP_POINTS,
    BAD_CAMERA_SCATTER_RATIO_THRESHOLD, BAD_CAMERA_MIN_CAMERAS,
    OFFSET_CAMERA_SIGMA_THRESHOLD,
    CATASTROPHIC_MIN_POINTS_PER_CAMERA, CATASTROPHIC_MAG_EXCURSION,
    CATASTROPHIC_SUPPORT_WINDOW_DAYS, CATASTROPHIC_SUPPORT_EXCURSION,
    CATASTROPHIC_MAX_FRACTION,
)

colors = ["#6b8bcd", "#b3b540", "#8f62ca", "#5eb550", "#c75d9c", "#4bb092", "#c5562f", "#6c7f39",
              "#ce5761", "#c68c45", '#b5b246', '#d77fcc', '#7362cf', '#ce443f', '#3fc1bf', '#cda735',
              '#a1b055']


def log(message: str, quiet: bool = False) -> None:
    """Print *message* to stdout unless *quiet* is True."""
    if not quiet:
        print(message, flush=True)


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


def clean_lc(df, max_error_absolute=CLEAN_LC_MAX_ERROR_ABSOLUTE, max_error_sigma=CLEAN_LC_MAX_ERROR_SIGMA):
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


def read_skypatrol_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read a SkyPatrol CSV and remap columns to the ASAS-SN-like schema."""
    csv_path = Path(csv_path)
    df = pd.read_csv(
        csv_path,
        comment="#",
        skip_blank_lines=True,
        dtype={
            "JD": float,
            "Flux": float,
            "Flux Error": float,
            "Mag": float,
            "Mag Error": float,
            "Limit": float,
            "FWHM": float,
            "Filter": "string",
            "Quality": "string",
            "Camera": "string",
        },
    )

    rename_map = {
        "Flux": "flux",
        "Flux Error": "flux_error",
        "Mag": "mag",
        "Mag Error": "error",
        "Limit": "limit",
        "FWHM": "fwhm",
        "Filter": "filter_band",
        "Quality": "quality_flag",
        "Camera": "camera",
    }
    df = df.rename(columns=rename_map)

    df["JD"] = pd.to_numeric(df["JD"], errors="coerce")
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    df["error"] = pd.to_numeric(df["error"], errors="coerce")

    if "flux" in df.columns:
        df["flux"] = pd.to_numeric(df["flux"], errors="coerce")
    else:
        df["flux"] = np.nan

    if "flux_error" in df.columns:
        df["flux_error"] = pd.to_numeric(df["flux_error"], errors="coerce")
    else:
        df["flux_error"] = np.nan

    df["camera"] = df["camera"].astype(str).str.strip()
    df["camera#"] = df["camera"]
    df["cam_field"] = df["camera#"]

    df["quality_flag"] = df["quality_flag"].astype(str).str.strip().str.upper()
    df["good_bad"] = (df["quality_flag"] == "G").astype(int)
    df["saturated"] = 0

    filt = df["filter_band"].astype(str).str.strip().str.lower()
    band_map = {"v": 1, "g": 0}
    df["v_g_band"] = filt.map(band_map)
    df = df[df["v_g_band"].notna()].copy()
    df["v_g_band"] = df["v_g_band"].astype(int)

    df = df[pd.notna(df["JD"]) & pd.notna(df["mag"])].copy()
    df = df.sort_values("JD").reset_index(drop=True)
    return df


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
    window_days: float = BAD_CAMERA_WINDOW_DAYS,
    min_overlap_points: int = BAD_CAMERA_MIN_OVERLAP_POINTS,
    scatter_ratio_threshold: float = BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    min_cameras_for_comparison: int = BAD_CAMERA_MIN_CAMERAS,
    cam_col: str = "camera#",
    t_col: str = "JD",
    mag_col: str = "mag",
) -> set:
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
    set
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
    bad_cameras: set = set()
    for cam, ratios in cam_scatter_ratios.items():
        if len(ratios) >= 3:  # Need at least 3 comparisons
            # Use median ratio to be robust to outliers
            median_ratio = np.median(ratios)
            if median_ratio > scatter_ratio_threshold:
                bad_cameras.add(cam)

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
                bad_cameras.add(cam)

    return bad_cameras


def identify_offset_cameras(
    df_lc: pd.DataFrame,
    *,
    window_days: float = BAD_CAMERA_WINDOW_DAYS,
    min_overlap_points: int = BAD_CAMERA_MIN_OVERLAP_POINTS,
    offset_sigma_threshold: float = OFFSET_CAMERA_SIGMA_THRESHOLD,
    min_cameras_for_comparison: int = BAD_CAMERA_MIN_CAMERAS,
    cam_col: str = "camera#",
    t_col: str = "JD",
    mag_col: str = "mag",
    remove_full_camera: bool = True,
) -> tuple[set, set[tuple]]:
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
    bad_cameras : set
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
    cam_offset_violations = {c: [] for c in cameras}

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
                cam_medians[cam] = np.median(cam_df[mag_col].dropna().values)

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
    bad_cameras: set = set()
    bad_windows: set[tuple] = set()

    for cam, violations in cam_offset_violations.items():
        if len(violations) >= 2:  # Need at least 2 violations to be confident
            if remove_full_camera:
                bad_cameras.add(cam)
            else:
                for start, end in violations:
                    bad_windows.add((cam, start, end))

    return bad_cameras, bad_windows


def identify_catastrophic_outlier_cameras(
    df_lc: pd.DataFrame,
    *,
    cam_col: str = "camera#",
    t_col: str = "JD",
    mag_col: str = "mag",
    min_points_per_camera: int = CATASTROPHIC_MIN_POINTS_PER_CAMERA,
    mag_excursion_threshold: float = CATASTROPHIC_MAG_EXCURSION,
    support_window_days: float = CATASTROPHIC_SUPPORT_WINDOW_DAYS,
    support_excursion_threshold: float = CATASTROPHIC_SUPPORT_EXCURSION,
    min_catastrophic_points: int = 1,
    max_catastrophic_fraction: float = CATASTROPHIC_MAX_FRACTION,
) -> set:
    """
    Identify cameras with isolated catastrophic magnitude excursions.

    A point is considered catastrophic for a camera when its deviation from the
    camera median is >= ``mag_excursion_threshold`` and nearby points from other
    cameras (within ``support_window_days``) do not show a comparable excursion.

    The camera is flagged if the number of such unsupported points is high enough
    to indicate instrument/systematic behavior, while still rare in recurrence.
    """
    if df_lc.empty or cam_col not in df_lc.columns:
        return set()

    finite = np.isfinite(df_lc[t_col]) & np.isfinite(df_lc[mag_col])
    df_use = df_lc.loc[finite, [cam_col, t_col, mag_col]].copy()
    if df_use.empty:
        return set()

    camera_medians = df_use.groupby(cam_col)[mag_col].median().to_dict()
    cameras = sorted(df_use[cam_col].dropna().unique())
    bad_cameras: set = set()

    for cam in cameras:
        cam_df = df_use[df_use[cam_col] == cam]
        n_cam = len(cam_df)
        if n_cam < int(min_points_per_camera):
            continue

        cam_med = camera_medians.get(cam, np.nan)
        if not np.isfinite(cam_med):
            continue

        cam_dev = np.abs(cam_df[mag_col].to_numpy(dtype=float) - float(cam_med))
        catastrophic_mask = cam_dev >= float(mag_excursion_threshold)
        if not np.any(catastrophic_mask):
            continue

        catastrophic_points = cam_df.loc[catastrophic_mask, [t_col, mag_col]]
        unsupported_count = 0

        for _, row in catastrophic_points.iterrows():
            t0 = float(row[t_col])
            others = df_use[(df_use[cam_col] != cam) & (np.abs(df_use[t_col] - t0) <= float(support_window_days))]

            if others.empty:
                unsupported_count += 1
                continue

            other_devs = []
            for other_cam, grp in others.groupby(cam_col):
                other_med = camera_medians.get(other_cam, np.nan)
                if not np.isfinite(other_med):
                    continue
                dev = np.abs(grp[mag_col].to_numpy(dtype=float) - float(other_med))
                if dev.size:
                    other_devs.append(float(np.nanmedian(dev)))

            if not other_devs:
                unsupported_count += 1
                continue

            if float(np.nanmax(other_devs)) < float(support_excursion_threshold):
                unsupported_count += 1

        frac = unsupported_count / float(n_cam)
        if (unsupported_count >= int(min_catastrophic_points)) and (frac <= float(max_catastrophic_fraction)):
            bad_cameras.add(cam)

    return bad_cameras


def filter_bad_cameras(
    df_lc: pd.DataFrame,
    raw2_df: pd.DataFrame | None = None,
    lc_path: str | None = None,
    *,
    filter_scatter: bool = True,
    filter_offset: bool = True,
    filter_catastrophic: bool = True,
    offset_sigma_threshold: float = 15.0,
    remove_full_camera: bool = True,
    **kwargs
) -> tuple[pd.DataFrame, set]:
    """
    Filter out cameras with anomalously high scatter or systematic offsets.

    Combines three filters:
    1. Scatter filter (identify_bad_cameras): flags cameras with high MAD scatter
    2. Offset filter (identify_offset_cameras): flags cameras with systematic median offsets
    3. Catastrophic outlier filter (identify_catastrophic_outlier_cameras):
       flags cameras with isolated 3+ mag excursions unsupported by other cameras

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
    filter_catastrophic : bool
        Apply isolated catastrophic-outlier filtering (default: True)
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
    bad_cameras : set
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
    bad_cameras: set = set()
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

    # 3. Catastrophic one-off outlier filter
    if filter_catastrophic:
        catastrophic_bad = identify_catastrophic_outlier_cameras(
            df_lc if df_filtered is df_lc else df_filtered,
            cam_col=cam_col,
            t_col=t_col,
            mag_col=kwargs.get("mag_col", "mag"),
            min_points_per_camera=kwargs.get("catastrophic_min_points", 30),
            mag_excursion_threshold=kwargs.get("catastrophic_mag_excursion", 3.0),
            support_window_days=kwargs.get("catastrophic_support_window_days", 2.0),
            support_excursion_threshold=kwargs.get("catastrophic_support_excursion", 0.75),
            min_catastrophic_points=kwargs.get("catastrophic_min_count", 1),
            max_catastrophic_fraction=kwargs.get("catastrophic_max_fraction", 0.03),
        )
        bad_cameras.update(catastrophic_bad)
    
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


# =============================================================================
# Shared rejection-logging utility
# =============================================================================

def log_rejections(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    filter_name: str,
    log_csv: "str | Path | None",
) -> None:
    """Log rejected candidates to a CSV file.

    Searches for an ID column in order: "ASAS-SN ID", "path", "asas_sn_id",
    "id", "source_id", then falls back to the first column.  Appends the full
    rows of rejected candidates plus a ``rejection_reason`` column.
    """
    if log_csv is None:
        return

    id_col = None
    for candidate in ["ASAS-SN ID", "path", "asas_sn_id", "id", "source_id"]:
        if candidate in df_before.columns:
            id_col = candidate
            break
    if id_col is None:
        id_col = df_before.columns[0]

    before_ids = set(df_before[id_col].astype(str))
    after_ids = set(df_after[id_col].astype(str))
    rejected_ids = before_ids - after_ids

    if not rejected_ids:
        return

    rejected = df_before[df_before[id_col].astype(str).isin(rejected_ids)].copy()
    rejected["rejection_reason"] = filter_name

    log_path = Path(log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = not log_path.exists() or log_path.stat().st_size == 0
    rejected.to_csv(log_path, mode="a", header=header, index=False)


# =============================================================================
# Shared Gaia TAP batch cone-search utility
# =============================================================================

def batch_gaia_cone_query(
    coords_df: pd.DataFrame,
    *,
    select_cols: str,
    extra_where: str = "",
    match_radius_arcsec: float,
    chunk_size: int,
    n_workers: int,
    verbose: bool = False,
) -> pd.DataFrame:
    """Batch Gaia TAP query using table upload for efficient cone search.

    Uses a pyvo async job with an uploaded coordinate table for server-side
    crossmatch.  ``coords_df`` must have columns ``_idx``, ``ra``, ``dec``.
    """
    import pyvo
    from astropy.table import Table
    from malca.config.config_paths import GAIA_AIP_TAP_URL

    if coords_df.empty:
        return pd.DataFrame()

    tap = pyvo.dal.TAPService(GAIA_AIP_TAP_URL)
    results = []
    chunks = [coords_df.iloc[i:i + chunk_size] for i in range(0, len(coords_df), chunk_size)]

    def process_chunk(chunk_df):
        try:
            upload_table = Table.from_pandas(chunk_df[["_idx", "ra", "dec"]])
            query = f"""
            SELECT
                u._idx as _idx,
                g.source_id,
                {select_cols},
                DISTANCE(POINT('ICRS', g.ra, g.dec), POINT('ICRS', u.ra, u.dec)) * 3600.0 as sep_arcsec
            FROM TAP_UPLOAD.upload_table AS u
            JOIN gaiadr3.gaia_source AS g
            ON 1=CONTAINS(
                POINT('ICRS', g.ra, g.dec),
                CIRCLE('ICRS', u.ra, u.dec, {match_radius_arcsec / 3600.0})
            )
            {extra_where}
            """
            result = tap.run_async(query, uploads={"upload_table": upload_table})
            return result.to_table().to_pandas() if result else pd.DataFrame()
        except Exception as e:
            if verbose:
                print(f"Gaia batch query error: {e}")
            return pd.DataFrame()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_chunk, chunk): i for i, chunk in enumerate(chunks)}
        for future in tqdm(
            as_completed(futures), total=len(futures),
            desc="Gaia batch query", disable=not verbose,
        ):
            result = future.result()
            if not result.empty:
                results.append(result)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def batch_tap_crossmatch(
    coords_df: pd.DataFrame,
    *,
    tap_url: str,
    catalog_table: str,
    select_cols: str,
    ra_col: str = "RAJ2000",
    dec_col: str = "DEJ2000",
    match_radius_arcsec: float = 3.0,
    chunk_size: int = 1000,
    n_workers: int = 4,
    verbose: bool = False,
    desc: str = "TAP crossmatch",

) -> pd.DataFrame:
    """Batch TAP crossmatch using coordinate upload.

    Generic utility for any TAP service that supports TAP_UPLOAD (VizieR,
    SIMBAD, etc).  ``coords_df`` must have columns ``_idx``, ``ra``, ``dec``.

    Returns a DataFrame with ``_idx``, the selected columns, and
    ``sep_arcsec``.  Callers should de-duplicate by ``_idx`` as needed.
    """
    from astroquery.utils.tap.core import TapPlus
    from astropy.table import Table

    if coords_df.empty:
        return pd.DataFrame()

    results: list[pd.DataFrame] = []
    chunks = [coords_df.iloc[i:i + chunk_size] for i in range(0, len(coords_df), chunk_size)]

    def process_chunk(chunk_df):
        try:
            tap = TapPlus(url=tap_url)
            upload_table = Table.from_pandas(chunk_df[["_idx", "ra", "dec"]])

            query = f"""
            SELECT
                u._idx AS _idx,
                {select_cols},
                DISTANCE(POINT('ICRS', c.{ra_col}, c.{dec_col}),
                         POINT('ICRS', u.ra, u.dec)) * 3600.0 AS sep_arcsec
            FROM TAP_UPLOAD.upload_table AS u
            JOIN {catalog_table} AS c
            ON 1=CONTAINS(
                POINT('ICRS', c.{ra_col}, c.{dec_col}),
                CIRCLE('ICRS', u.ra, u.dec, {match_radius_arcsec / 3600.0})
            )
            """

            job = tap.launch_job_async(
                query,
                upload_resource=upload_table,
                upload_table_name="upload_table",
                verbose=False,
            )
            result = job.get_results()
            return result.to_pandas() if result is not None and len(result) > 0 else pd.DataFrame()
        except Exception as e:
            if verbose:
                print(f"  {desc} chunk error: {e}")
            return pd.DataFrame()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_chunk, chunk): i for i, chunk in enumerate(chunks)}

        for future in tqdm(
            as_completed(futures), total=len(futures),
            desc=desc, disable=not verbose,
        ):
            result = future.result()
            if not result.empty:
                results.append(result)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)
