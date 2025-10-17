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
    """adopted from Brayden's code; boolean flag for box filtering"""

    mag = np.asarray(df["mag"], float)
    jd = np.asarray(df["JD"], float)

    meanmag = mag.mean()
    df_mag_avg = mag - meanmag

    peak, _ = scipy.signal.find_peaks(
        df_mag_avg,
        prominence=prominence,
        distance=distance,
        height=height,
        width=width,
    )

    n_peaks = len(peak)

    if apply_box_filter:
        jd_span = float(jd[-1] - jd[0]) if jd.size > 1 else 0.0
        peaks_per_time = (n_peaks / jd_span) if jd_span > 0 else np.inf
        std_mag = float(np.nanstd(mag))

        if (
            n_peaks == 0
            or n_peaks >= max_dips
            or peaks_per_time > max_peaks_per_time
            or std_mag > max_std
        ):
            return pd.Series(dtype=int, name="peaks"), meanmag, 0

    return pd.Series(peak, name="peaks"), meanmag, n_peaks



def peak_search():

    pass