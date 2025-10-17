
import os

import numpy as np
import pandas as pd
import scipy.signal


# all of this code is adapted from Brayden JoHantgen's code



def read_lightcurve_dat(asas_sn_id, guide = 'known_dipper_lightcurves/'):
    """
    Input: 
        asas_sn_id: the asassn id of the desired star
        guide: the path to the data file of the desired star

    Output: 
        dfv: This is the dataframe for the V-band data of the star
        dfg: This is the dataframe for the g-band data of the star
    
    This function reads the data of the desired star by going to the corresponding file and copying the data of that file onto 
    a data frame. This data frame is then sorted into two data frames by comparing the value in the Photo filter column. If the
    Photo filter column data has a value of one, its row is sorted into the data frame corresponding to the V-band. If the Photo
    filter column data has a value of zero, it gets sorted into the data frame corresponding to the g-band.
    """
    fname = os.path.join(guide, str(asas_sn_id)+'.dat')

    dfv = pd.DataFrame()
    dfg = pd.DataFrame()

    fdata = pd.read_fwf(fname, header=None)
    fdata.columns = ["JD", "Mag", "Mag_err", "Quality", "Cam_number", "Phot_filter", "Camera"] #These are the columns of data

    dfv = fdata.loc[fdata["Phot_filter"] == 1].reset_index(drop=True) #This sorts the data into the V-band
    dfg = fdata.loc[fdata["Phot_filter"] == 0].reset_index(drop=True) #This sorts the data into the g-band

    dfv['Mag'].astype(float)
    dfg['Mag'].astype(float)

    dfv['JD'].astype(float)
    dfg['JD'].astype(float)

    return dfv, dfg


def read_lightcurve_csv(asas_sn_id, guide = 'known_dipper_lightcurves/'):
    """
    Input: 
        asas_sn_id: the asassn id of the desired star
        guide: the path to the data file of the desired star

    Output: 
        dfv: This is the dataframe for the V-band data of the star
        dfg: This is the dataframe for the g-band data of the star
    
    This function reads the data of the desired star by going to the corresponding file and copying the data of that file onto 
    a data frame. This data frame is then sorted into two data frames by comparing the value in the Photo filter column. If the
    Photo filter column data has a value of one, its row is sorted into the data frame corresponding to the V-band. If the Photo
    filter column data has a value of zero, it gets sorted into the data frame corresponding to the g-band.
    """
    fname = os.path.join(guide, str(asas_sn_id)+'.csv')

    df = pd.read_csv(fname)

    df['Mag'] = pd.to_numeric(df['mag'],errors='coerce')
    df = df.dropna()
    
    df['Mag'].astype(float)
    df['JD'] = df.HJD.astype(float)

    dfg = df.loc[df['Filter'] == 'g'].reset_index(drop=True)
    dfv = df.loc[df['Filter'] == 'V'].reset_index(drop=True)

    return dfv, dfg

# This function finds the peaks 
def find_peak(
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
    """Locate peaks in a light curve and optionally apply the historical "box" filter.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ``Mag`` and ``JD``.
    prominence, distance, height, width : float
        Forwarded to :func:`scipy.signal.find_peaks`.
    apply_box_filter : bool, optional
        If True (default), enforce the legacy box criteria used in the original
        pipeline (number of dips, peaks per unit time, and scatter limits).
    max_dips : int, optional
        Maximum allowed number of dips to pass the filter.
    max_std : float, optional
        Maximum allowed standard deviation of the magnitudes.
    max_peaks_per_time : float, optional
        Maximum allowed peak density (peaks per unit JD span).

    Returns
    -------
    pandas.Series, float, int
        The peak indices, the mean magnitude, and the number of detected peaks.
        If the box filter rejects the light curve, an empty Series and zero count
        are returned (mean magnitude is still reported).
    """

    if df.empty:
        return pd.Series(dtype=int), np.nan, 0

    work = df.copy()
    work["Mag"] = pd.to_numeric(work["Mag"], errors="coerce")
    work["JD"] = pd.to_numeric(work["JD"], errors="coerce")
    work = work.dropna(subset=["Mag", "JD"]).reset_index(drop=True)
    if work.empty:
        return pd.Series(dtype=int), np.nan, 0

    mag = work["Mag"].to_numpy(dtype=float)
    jd = work["JD"].to_numpy(dtype=float)

    meanmag = float(np.nanmean(mag))
    df_mag_avg = mag - meanmag

    peaks, _ = scipy.signal.find_peaks(
        df_mag_avg,
        prominence=prominence,
        distance=distance,
        height=height,
        width=width,
    )

    peaks = peaks.astype(int)
    length = int(peaks.size)

    if apply_box_filter:
        jd_span = float(jd[-1] - jd[0]) if jd.size > 1 else 0.0
        peaks_per_time = (length / jd_span) if jd_span > 0 else np.inf
        std_mag = float(np.nanstd(mag))

        if (
            length == 0
            or length >= max_dips
            or peaks_per_time > max_peaks_per_time
            or std_mag > max_std
        ):
            return pd.Series(dtype=int), meanmag, 0

    peak_series = pd.Series(peaks, name="peaks")
    return peak_series, meanmag, length
# End of the find_peak

# This function creates a custom id using the position of the star

# End of custom_id


# This function plots the light curve

# End of plot_light_curve

#

#

#

#

#
#def peak_params(df):
#
