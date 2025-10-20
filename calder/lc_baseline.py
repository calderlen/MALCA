import numpy as np
import pandas as pd
#from celerite2 import terms, GaussianProcess
#from scipy.optimize import minimize

# --- helpers ---
def rolling_time_median(jd, mag, days=300.0, min_points=10, min_days=30.0):
    """
    Rolling median in time (default 300 d). Requires at least min_points finite magnitudes within the window; halves the window down to min_days
    """

    jd = np.asarray(jd, float)
    mag = np.asarray(mag, float)

    out = np.full_like(mag, np.nan, dtype=float)
    for i, t0 in enumerate(jd):
        window = float(days)
        while window >= float(min_days):
            mask = (jd >= t0 - window / 2.0) & (jd <= t0 + window / 2.0)
            vals = mag[mask]
            good = np.isfinite(vals)
            if int(good.sum()) >= int(min_points):
                out[i] = np.nanmedian(vals[good])
                break
            window /= 2.0  # halve the window and try again
    return out

# --- baseline functions ---
def global_mean_baseline(
    df,
    t_col="JD",         # this input is unnecessary, but just copying schema of per_camera_median_baseline for now
    mag_col="mag",
    err_col="error",
):
    df_out = df.copy()
    for col in ("baseline", "resid", "sigma_resid"):
        if col not in df_out.columns:
            df_out[col] = np.nan

    m = df_out.loc[:, mag_col].to_numpy(dtype=float)
    e = df_out.loc[:, err_col].to_numpy(dtype=float)

    baseline = np.full_like(m, np.nan, dtype=float)
    resid = np.full_like(m, np.nan, dtype=float)

    good = np.isfinite(m)
    if good.any():
        mean_mag = float(np.mean(m[good]))
        baseline[:] = mean_mag
        resid = m - mean_mag

    resid_good = np.isfinite(resid)
    if resid_good.any():
        resid_vals = resid[resid_good]
        med_resid = float(np.median(resid_vals))
        mad = float(1.4826 * np.median(np.abs(resid_vals - med_resid)))
    else:
        med_resid = np.nan
        mad = np.nan

    e_good = np.isfinite(e)
    e_med = float(np.median(e[e_good])) if e_good.any() else np.nan

    mad_num = mad if np.isfinite(mad) else 0.0
    e_med_num = e_med if np.isfinite(e_med) else 0.0
    robust_std = float(np.sqrt(mad_num**2 + e_med_num**2))
    robust_std = max(robust_std, 1e-6)

    sigma_resid = resid / robust_std

    df_out.loc[:, "baseline"] = baseline
    df_out.loc[:, "resid"] = resid
    df_out.loc[:, "sigma_resid"] = sigma_resid

    return df_out


def per_camera_mean_baseline(
    df,
    t_col="JD",         # this input is unnecessary, but just copying schema of per_camera_median_baseline for now
    mag_col="mag",
    err_col="error",
    cam_col="camera#",
):
    df_out = df.copy()
    for col in ("baseline", "resid", "sigma_resid"):
        if col not in df_out.columns:
            df_out[col] = np.nan

    for _, sub in df_out.groupby(cam_col, group_keys=False):
        idx = sub.index

        m = df_out.loc[idx, mag_col].to_numpy(dtype=float)
        e = df_out.loc[idx, err_col].to_numpy(dtype=float)

        baseline = np.full_like(m, np.nan, dtype=float)
        resid = np.full_like(m, np.nan, dtype=float)

        good = np.isfinite(m)
        if good.any():
            cam_mean = float(np.mean(m[good]))
            baseline[:] = cam_mean
            resid = m - cam_mean

        resid_good = np.isfinite(resid)
        if resid_good.any():
            resid_vals = resid[resid_good]
            med_resid = float(np.median(resid_vals))
            mad = float(1.4826 * np.median(np.abs(resid_vals - med_resid)))
        else:
            med_resid = np.nan
            mad = np.nan

        e_good = np.isfinite(e)
        e_med = float(np.median(e[e_good])) if e_good.any() else np.nan

        mad_num = mad if np.isfinite(mad) else 0.0
        e_med_num = e_med if np.isfinite(e_med) else 0.0
        robust_std = float(np.sqrt(mad_num**2 + e_med_num**2))
        robust_std = max(robust_std, 1e-6)

        sigma_resid = resid / robust_std

        df_out.loc[idx, "baseline"] = baseline
        df_out.loc[idx, "resid"] = resid
        df_out.loc[idx, "sigma_resid"] = sigma_resid

    return df_out

def per_camera_median_baseline(
    df,
    days=300.0,
    min_points=10,
    t_col="JD",
    mag_col="mag",
    err_col="error",
    cam_col="camera#",
):
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
        resid_good = np.isfinite(resid)
        if resid_good.any():
            resid_vals = resid[resid_good]
            med_resid = float(np.median(resid_vals))
            mad = float(1.4826 * np.median(np.abs(resid_vals - med_resid)))
        else:
            med_resid = np.nan
            mad = np.nan

        e_good = np.isfinite(e)
        e_med = float(np.median(e[e_good])) if e_good.any() else np.nan

        mad_num = mad if np.isfinite(mad) else 0.0
        e_med_num = e_med if np.isfinite(e_med) else 0.0
        robust_std = float(np.sqrt(mad_num**2 + e_med_num**2))
        robust_std = max(robust_std, 1e-6)  # avoid 0/NaN

        sigma_resid = resid / robust_std

        df_out.loc[idx, "baseline"] = base
        df_out.loc[idx, "resid"] = resid
        df_out.loc[idx, "sigma_resid"] = sigma_resid

    return df_out








#def fit_gp_baseline(
#    jd,
#    mag,
#    error,
#    *,
#    kernel="matern32
#    sho_period = None,      # period of SHO
#    sho_Q = 1.0,            # quality factor of SHO
#    mean_mag = None,
#    mask = None,            # this would be a boolean mask that filters out known dips
#    jitter_init = 1.0,    
#):
    


#def fit_gp_baseline(
#        jd,
#        mag,
#        error, 
#):
#
#    jd, mag, error = data

#    # quasi-periodic term
#    term1 = terms.SHOTerm(sigma, rho, tau)
    
#    # jitter term
#    term2 = 

#
#
#
#    # maximize the likelihood function for the parameters of the kernel: mean, jitter, 
#    
#
#    gp = celerite2.GaussianProcess(kernel, mean=)
#    gp.compute(jd, yerr=error)
#
#    pass
