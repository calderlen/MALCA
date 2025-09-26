import numpy as np
import pandas as pd
from utils import read_lc_dat

# need to get the vsx catalog
# get the asassn id of each surviving lc after crossmatchign with vsx
# load the asassn_id's into a a list
# then for all the asassn_id's, append the v and g mag lightcurves to them
# probably need to rewrite read_lightcurve to account for this

#df_v, df_g = read_lightcurve(asassn_id, path)


def rolling_time_median(jd, mag, days=30., min_points=10):
    '''
    rolling median in time (currently 2 months), with at least min_points in the window to compute median
    '''
    
    t = np.asarray(jd)
    mag = np.asarray(mag)

    mag_med = np.full_like(mag, np.nan, dtype=float)

    for i, ti in enumerate(t):
        mask = (t >= ti - days/2) & (t <= ti + days/2)
        if np.sum(mask) >= min_points:
            mag_med[i] = np.median(mag[mask])

    # fill nearest
    if np.any(np.isnan(mag_med)):
        mask = ~np.isnan(mag_med)
        full_med = np.interp(t, t[mask], mag_med[mask])
        mag_med = np.where(np.isnan(mag_med), full_med, mag_med)

    # if any NaNs in full_med
    
    # if there are not enough points in the window, fill with global median
    global_med = np.nanmedian(mag[mask])

    return np.where(np.isnan(mag_med), global_med, mag_med)


def per_camera_baseline(df, days=30., min_points=10, t_col="JD", mag_col="mag", err_col="error", cam_col="camera#"):
    
    # work on a copy; initialize outputs
    out = df.copy()
    for col in ("baseline", "resid", "z"):
        if col not in out.columns:
            out[col] = np.nan

    # group by camera and fill columns
    for _, sub in out.groupby(cam_col, group_keys=False):
        idx = sub.index

        t = out.loc[idx, t_col].to_numpy(dtype=float)
        m = out.loc[idx, mag_col].to_numpy(dtype=float)
        e = out.loc[idx, err_col].to_numpy(dtype=float)

        base = rolling_time_median(t, m, days=days, min_points=min_points)
        resid = m - base

        # robust scale using MAD and include typical photometric error
        med_resid = np.nanmedian(resid)
        mad = 1.4826 * np.nanmedian(np.abs(resid - med_resid))
        e_med = np.nanmedian(e)

        robust_std = np.sqrt(mad**2 + e_med**2)
        robust_std = max(float(robust_std), 1e-6)  # avoid 0/NaN

        z = resid / robust_std

        out.loc[idx, "baseline"] = base
        out.loc[idx, "resid"] = resid
        out.loc[idx, "z"] = z

    return out
