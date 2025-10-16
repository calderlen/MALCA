import numpy as np
import pandas as pd

# need to get the vsx catalog
# get the asassn id of each surviving lc after crossmatchign with vsx
# load the asassn_id's into a a list
# then for all the asassn_id's, append the v and g mag lightcurves to them
# probably need to rewrite read_lightcurve to account for this

#df_v, df_g = read_lightcurve(asassn_id, path)


def rolling_time_median(jd, mag, days=30., min_points=10):
    """
    rolling median in time (currently 2 months), with at least min_points in the window to compute median
    """
    
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
    """
    returns a df that mirrors input df but with three extra float columns: (1) baseline, a rolling 30-day median mag computed within each camera group; (2) resid, residual mag-baseline per-camera; (3) sigma_resid, residual divided by (MAD+mag_error) in quadrature, yielding a per-point significance
    """
    # work on a copy; initialize df_outputs
    df_out = df.copy()
    for col in ("baseline", "resid", "sigma_resid"):
        if col not in df_out.columns:
            df_out[col] = np.nan

    # group by camera and fill columns
    for _, sub in df_out.groupby(cam_col, group_keys=False):
        idx = sub.index

        t = df_out.loc[idx, t_col].to_numpy(dtype=float)
        m = df_out.loc[idx, mag_col].to_numpy(dtype=float)
        e = df_out.loc[idx, err_col].to_numpy(dtype=float)

        base = rolling_time_median(t, m, days=days, min_points=min_points)
        resid = m - base

        # robust scatter
        med_resid = np.nanmedian(resid)
        mad = 1.4826 * np.nanmedian(np.abs(resid - med_resid))
        e_med = np.nanmedian(e)

        robust_std = np.sqrt(mad**2 + e_med**2)
        robust_std = max(float(robust_std), 1e-6)  # avoid 0/NaN

        sigma_resid = resid / robust_std

        df_out.loc[idx, "baseline"] = base
        df_out.loc[idx, "resid"] = resid
        df_out.loc[idx, "sigma_resid"] = sigma_resid

    return df_out
