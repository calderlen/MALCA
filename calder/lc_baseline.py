import numpy as np
import pandas as pd
#from celerite2 import terms, GaussianProcess
#from scipy.optimize import minimize


def global_mean_baseline(
    df,
    t_col="JD",         
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


def global_median_baseline(
    df,
    t_col="JD",         
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
        median_mag = float(np.median(m[good]))
        baseline[:] = median_mag
        resid = m - median_mag

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


# rolling time window helper functions for subsequent baselines


def rolling_time_median(jd, mag, days=300.0, min_points=10, min_days=30.0, past_only=True):
    """
    Rolling median in time using pure numpy optimization (searchsorted).
    If past_only=True, uses [t0 - days, t0] (one-sided) to avoid future-leakage into ongoing dips.
    Halves 'days' down to min_days until >= min_points exist.
    """

    n = len(jd)
    out = np.full(n, np.nan, dtype=float)
    
    # cast to numpy arrays just in case
    jd = np.asarray(jd, dtype=float)
    mag = np.asarray(mag, dtype=float)

    for i in range(n):
        t0 = jd[i]
        window = float(days)
        
        while window >= float(min_days):
            if past_only:
                lo_val, hi_val = t0 - window, t0
            else:
                half = window / 2.0
                lo_val, hi_val = t0 - half, t0 + half
                
            idx_start = np.searchsorted(jd, lo_val, side='left')
            idx_end = np.searchsorted(jd, hi_val, side='right')
            
            # Slice the arrays (very fast view, no copy)
            vals = mag[idx_start:idx_end]
            
            # Check for NaNs
            # (In pure numpy, we must filter them explicitly before median)
            finite_vals = vals[np.isfinite(vals)]
            
            if len(finite_vals) >= int(min_points):
                out[i] = np.median(finite_vals)
                break
            
            window /= 2.0
            
    return out


def rolling_time_mad(jd, resid, days=200.0, min_points=10, min_days=20.0, past_only=True, add_err=None):
    """
    Rolling robust scatter (MAD) using pure NumPy optimization.
    1.4826 * median(|resid - median(resid)|)
    """
    n = len(jd)
    out = np.full(n, np.nan, dtype=float)
    
    jd = np.asarray(jd, dtype=float)
    resid = np.asarray(resid, dtype=float)
    
    # Handle add_err (scalar or array)
    if add_err is not None:
        if np.ndim(add_err) > 0:
            err_is_array = True
            err_array = np.asarray(add_err, dtype=float)
        else:
            err_is_array = False
            err_scalar = float(add_err)
    
    for i in range(n):
        t0 = jd[i]
        window = float(days)
        
        while window >= float(min_days):
            if past_only:
                lo_val, hi_val = t0 - window, t0
            else:
                half = window / 2.0
                lo_val, hi_val = t0 - half, t0 + half
            
            # BINARY SEARCH OPTIMIZATION
            idx_start = np.searchsorted(jd, lo_val, side='left')
            idx_end = np.searchsorted(jd, hi_val, side='right')
            
            vals = resid[idx_start:idx_end]
            finite_vals = vals[np.isfinite(vals)]
            
            if len(finite_vals) >= int(min_points):
                med = np.median(finite_vals)
                # MAD calculation
                mad = 1.4826 * np.median(np.abs(finite_vals - med))
                
                if add_err is not None:
                    if err_is_array:
                        err_here = err_array[i]
                    else:
                        err_here = err_scalar
                    mad = np.sqrt(mad**2 + err_here**2)
                
                out[i] = max(mad, 1e-6)
                break
            
            window /= 2.0
            
    return out


def global_rolling_median_baseline(
    df,
    days=1000.,
    min_points=10,
    t_col="JD",
    mag_col="mag",
    err_col="error",
):
    """
    A global rolling-median baseline (not per camera). Applies rolling_time_median
    to the entire dataset once, returning baseline/resid/sigma_resid columns.
    """
    df_out = df.copy()
    for col in ("baseline", "resid", "sigma_resid"):
        if col not in df_out.columns:
            df_out[col] = np.nan

    t = df_out.loc[:, t_col].to_numpy(dtype=float)
    m = df_out.loc[:, mag_col].to_numpy(dtype=float)
    e = df_out.loc[:, err_col].to_numpy(dtype=float)

    base = rolling_time_median(t, m, days=days, min_points=min_points)
    resid = m - base

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

    df_out.loc[:, "baseline"] = base
    df_out.loc[:, "resid"] = resid
    df_out.loc[:, "sigma_resid"] = sigma_resid

    return df_out


def global_rolling_mean_baseline(
    df,
    days=1000.,
    min_points=10,
    t_col="JD",
    mag_col="mag",
    err_col="error",
):
    """
    Similar to global_rolling_median_baseline but uses a rolling mean instead of median.
    """
    df_out = df.copy()
    for col in ("baseline", "resid", "sigma_resid"):
        if col not in df_out.columns:
            df_out[col] = np.nan

    t = df_out.loc[:, t_col].to_numpy(dtype=float)
    m = df_out.loc[:, mag_col].to_numpy(dtype=float)
    e = df_out.loc[:, err_col].to_numpy(dtype=float)

    baseline = np.full_like(m, np.nan, dtype=float)
    order = np.argsort(t)
    t_sorted = t[order]
    m_sorted = m[order]

    for idx_sorted, i in enumerate(order):
        t0 = t_sorted[idx_sorted]
        window = float(days)
        while window >= min_points:
            lo = t0 - window if True else t0 - window / 2.0
            hi = t0
            start = np.searchsorted(t_sorted, lo, side="left")
            end = np.searchsorted(t_sorted, hi, side="right")
            vals = m_sorted[start:end]
            finite = vals[np.isfinite(vals)]
            if finite.size >= min_points:
                baseline[i] = float(np.mean(finite))
                break
            window /= 2.0

    resid = m - baseline

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
    t_col="JD",         
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


def per_camera_trend_baseline(
        df,
        days_short=50., 
        days_long=800.0, 
        min_points=10, 
        last_window_guard=120.0, 
        t_col="JD", 
        mag_col="mag", 
        err_col="error", 
        cam_col="camera#"):
    """
    Multi-scale, one-sided (past-only) rolling-median baseline per camera to avoid
    future-leakage; rolling MAD for local significance; late-window guard for right-censored dips.
    """

    df_out = df.copy()
    for col in ("baseline", "resid", "sigma_resid"):
        if col not in df_out.columns:
            df_out[col] = np.nan

    # process per camera, sorted by time
    for _, sub in df_out.groupby(cam_col, group_keys=False):
        idx = sub.sort_values(t_col).index
        t = df_out.loc[idx, t_col].to_numpy(float)
        m = df_out.loc[idx, mag_col].to_numpy(float)
        e = df_out.loc[idx, err_col].to_numpy(float)

        # multi-scale, past-only baselines
        base_s = rolling_time_median(t, m, days=days_short, min_points=min_points, past_only=True)
        base_l = rolling_time_median(t, m, days=days_long, min_points=min_points, past_only=True)

        # choose baseline adaptively: prefer long if |base_s - base_l| is small (slow trend), else short
        choose_short = np.isfinite(base_s) & np.isfinite(base_l) & (np.abs(base_s - base_l) > 0.05) # 0.05 mag heuristic
        baseline = np.where(choose_short & np.isfinite(base_s), base_s,
        np.where(np.isfinite(base_l), base_l, base_s))

        resid = m - baseline

        # rolling local robust scatter; add median(err) in quadrature
        e_med = np.nanmedian(e) if np.isfinite(e).any() else 0.0
        sigma_loc = rolling_time_mad(t, resid, days=days_short, min_points=max(8, min_points//2),
        past_only=True, add_err=e_med)

        # late-window guard: within last N days, allow right-censored dips (don’t inflate sigma by requiring return)
        tmax = np.nanmax(t)
        near_end = (tmax - t) <= float(last_window_guard)
        
        # If sigma_loc missing near the end, fall back to global robust + e_med
        if np.isnan(sigma_loc[near_end]).any():
            r_good = np.isfinite(resid)
            if r_good.any():
                med_r = np.nanmedian(resid[r_good])
                mad_r = 1.4826 * np.nanmedian(np.abs(resid[r_good] - med_r))
                robust = float(np.sqrt(max(mad_r, 0.0)**2 + max(e_med, 0.0)**2))
                sigma_loc[near_end & ~np.isfinite(sigma_loc)] = max(robust, 1e-6)

        sigma_loc = np.where(np.isfinite(sigma_loc), sigma_loc, 1e-6)
        sigma_resid = resid / sigma_loc

        df_out.loc[idx, "baseline"] = baseline
        df_out.loc[idx, "resid"] = resid
        df_out.loc[idx, "sigma_resid"] = sigma_resid

    return df_out