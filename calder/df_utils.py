import numpy as np
import pandas as pd
import scipy


def naive_peak_search(
    df,
    prominence=0.17,
    distance=25,
    height=0.3,
    width=2,
    apply_box_filter=True,
    max_dips=10,
    max_std=0.15,
    max_peaks_per_time=0.015,
):
    """
    Peak finder that prefers per-camera-baseline residuals when available.
    Falls back to (mag - mean) if 'resid' is absent.
    """

    mag = np.asarray(df["mag"], float)
    jd = np.asarray(df["JD"], float)
    meanmag = float(np.nanmean(mag)) if mag.size else np.nan

    if "resid" in df.columns:
        resid = np.asarray(df["resid"], float)
        values = np.nan_to_num(resid, nan=0.0)
    else:
        values = mag - meanmag

    peak, _ = scipy.signal.find_peaks(
        values,
        prominence=prominence,
        distance=distance,
        height=height,
        width=width,
    )

    n_peaks = len(peak)

    if apply_box_filter:
        jd_span = float(jd[-1] - jd[0]) if jd.size > 1 else 0.0
        peaks_per_time = (n_peaks / jd_span) if jd_span > 0 else np.inf
        std_mag = float(np.nanstd(values))

        if (
            n_peaks == 0
            or n_peaks >= max_dips
            or peaks_per_time > max_peaks_per_time
            or std_mag > max_std
        ):
            return pd.Series(dtype=int, name="peaks"), meanmag, 0

    return pd.Series(peak, name="peaks"), meanmag, n_peaks



def peak_search(
    df
):
    """
    need to create GP baseline first before starting on this
    """
    

    pass
