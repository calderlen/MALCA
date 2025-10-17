import numpy as np
import pandas as pd

# need to get the vsx catalog
# get the asassn id of each surviving lc after crossmatchign with vsx
# load the asassn_id's into a a list
# then for all the asassn_id's, append the v and g mag lightcurves to them
# probably need to rewrite read_lightcurve to account for this

#df_v, df_g = read_lightcurve(asassn_id, path)


def rolling_time_median(jd, mag, days=300., min_points=10, min_days=30.):
    """
    rolling median in time (currently 300d), with at least min_points in the window to compute median
    """
    
    jd = np.asarray(jd, float)
    mag = np.asarray(mag, float)

    out = np.full_like(mag, np.nan, dtype=float)
    for i, t0 in enumerate(jd):
        window = days
        while window >= min_days:
            mask = (jd >= t0 - window/2) & (jd <= t0 + window/2)
            if mask.sum() >= min_points:
                out[i] = np.nanmedian(mag[mask])
                break
            window /= 2  # halve the window and try again
    return out


def per_camera_baseline(df, days=300., min_points=10, t_col="JD", mag_col="mag", err_col="error", cam_col="camera#"):
    """
    returns a df that mirrors input df but with three extra float columns: (1) baseline, a rolling 300-day median mag computed within each camera group; (2) resid, residual mag-baseline per-camera; (3) sigma_resid, residual divided by (MAD+mag_error) in quadrature, yielding a per-point significance
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
