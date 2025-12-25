import numpy as np
import pandas as pd
import warnings
from celerite2 import GaussianProcess, terms

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


def per_camera_gp_baseline(
    df,
    *,
    sigma=None,
    rho=None,
    q=0.7,
    S0=None,
    w0=None,
    jitter=0.006,
    t_col="JD",
    mag_col="mag",
    err_col="error",
    cam_col="camera#",
):
    """
    per-camera GP baseline (fixed SHO kernel)
    
    Supports two parameterizations:
    - sigma, rho, Q (default)
    - S0, w0, Q (alternative)
    """
    df_out = df.copy()
    for col in ("baseline", "resid", "sigma_resid"):
        if col not in df_out.columns:
            df_out[col] = np.nan

    for _, sub in df_out.groupby(cam_col, group_keys=False):
        idx = sub.sort_values(t_col).index
        t = df_out.loc[idx, t_col].to_numpy(dtype=float)
        y = df_out.loc[idx, mag_col].to_numpy(dtype=float)
        if err_col in df_out.columns:
            yerr = df_out.loc[idx, err_col].to_numpy(dtype=float)
        else:
            yerr = np.full_like(y, np.nan, dtype=float)

        finite = np.isfinite(t) & np.isfinite(y)
        if finite.sum() < 5:
            continue

        finite_idx = np.flatnonzero(finite)
        t_fit = t[finite_idx]
        y_fit = y[finite_idx]
        y_mean = np.mean(y_fit)
        y_centered = y_fit - y_mean
        yerr_fit = yerr[finite_idx]
        if not np.isfinite(yerr_fit).any():
            yerr_fit = np.full_like(y_fit, jitter, dtype=float)
        else:
            yerr_fit = np.where(np.isfinite(yerr_fit), yerr_fit, np.nanmedian(yerr_fit))
            yerr_fit = np.nan_to_num(yerr_fit, nan=jitter, posinf=jitter, neginf=jitter)

        # Build kernel - use S0, w0 if provided, otherwise use sigma, rho
        if S0 is not None and w0 is not None:
            k = terms.SHOTerm(S0=S0, w0=w0, Q=q)
        else:
            # Default to sigma, rho if not provided
            if sigma is None:
                sigma = 0.05
            if rho is None:
                rho = 200.0
            k = terms.SHOTerm(sigma=sigma, rho=rho, Q=q)

        try:
            gp = GaussianProcess(k)
            try:
                gp.compute(t_fit, diag=yerr_fit**2)
            except TypeError:
                # Fallback for celerite2 versions expecting keyword-only diag/yerr
                gp.compute(t_fit, diag=yerr_fit**2)
            mu, var = gp.predict(y_centered, t, return_var=True)
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"GP fit failed for camera group; leaving NaNs. Error: {exc}")
            continue

        baseline = np.asarray(mu, dtype=float) + y_mean
        resid = y - baseline

        # Use predicted variance + median error to get a scale for sigma_resid.
        var = np.asarray(var, dtype=float)
        med_err = float(np.nanmedian(yerr_fit)) if np.isfinite(yerr_fit).any() else 0.0
        scale = np.sqrt(np.maximum(var, 0.0) + med_err**2)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, med_err)
        sigma_resid = resid / scale

        df_out.loc[idx, "baseline"] = baseline
        df_out.loc[idx, "resid"] = resid
        df_out.loc[idx, "sigma_resid"] = sigma_resid

    return df_out


def per_camera_gp_baseline_masked(
    df,
    *,
    dip_sigma_thresh=-1.0,
    pad_days=100.0,

    # SHO kernel parameters (default, preferred)
    S0=0.0005,
    w0=0.0031415926535897933,
    Q=0.7,

    # RealTerm (OU mixture) parameters (optional, for backward compatibility)
    a1=None,
    rho1=None,
    a2=None,
    rho2=None,

    jitter=0.006,
    use_yerr=True,

    t_col="JD",
    mag_col="mag",
    err_col="error",
    cam_col="camera#",

    min_gp_points=10,
):
    """
    per-camera GP baseline with masking (dips excluded from fit)
    
    Masks out significant dips (thresholded by local MAD) before fitting the GP
    baseline to ensure the baseline follows the quiescent state rather than the dips.
    
    Supports two kernel parameterizations:
    - SHO kernel (default): S0, w0, Q
    - RealTerm (OU mixture): a1, rho1, a2, rho2 (for backward compatibility)
    
    If RealTerm parameters are explicitly provided, they take precedence.
    """
    df_out = df.copy()
    for col in ("baseline", "resid", "sigma_resid"):
        if col not in df_out.columns:
            df_out[col] = np.nan

    for _, sub in df_out.groupby(cam_col, group_keys=False):
        idx = sub.sort_values(t_col).index
        t = df_out.loc[idx, t_col].to_numpy(float)
        y = df_out.loc[idx, mag_col].to_numpy(float)

        if use_yerr and err_col in df_out.columns:
            yerr = df_out.loc[idx, err_col].to_numpy(float)
        else:
            yerr = np.full_like(y, np.nan, dtype=float)

        finite = np.isfinite(t) & np.isfinite(y)

        # --- 1) Cheap baseline: per-camera median
        y_med = float(np.nanmedian(y[finite]))
        r0 = y - y_med

        # --- 2) Robust global scale for masking
        r0_f = r0[finite]
        med_r = float(np.nanmedian(r0_f))
        mad_r = 1.4826 * float(np.nanmedian(np.abs(r0_f - med_r)))

        if use_yerr and np.isfinite(yerr).any():
            e_med = float(np.nanmedian(yerr[finite & np.isfinite(yerr)]))
        else:
            e_med = float(jitter)

        s0 = float(np.sqrt(max(mad_r, 0.0)**2 + max(e_med, 0.0)**2))
        s0 = max(s0, 1e-6)

        sig0 = r0 / s0
        dip_flag = finite & np.isfinite(sig0) & (sig0 < float(dip_sigma_thresh))

        # --- 3) Pad mask in time around dip candidates
        keep = finite.copy()
        if dip_flag.any():
            t_dip = t[dip_flag]
            bad = np.zeros_like(keep, dtype=bool)
            for td in t_dip:
                bad |= (np.abs(t - td) <= float(pad_days))
            keep &= ~bad

        if keep.sum() < min_gp_points:
            baseline = np.full_like(y, y_med, dtype=float)
            resid = y - baseline
            df_out.loc[idx, "baseline"] = baseline
            df_out.loc[idx, "resid"] = resid
            df_out.loc[idx, "sigma_resid"] = resid / s0
            continue

        t_fit = t[keep]
        y_fit = y[keep]

        if use_yerr and np.isfinite(yerr[keep]).any():
            yerr_fit = yerr[keep]
            med = float(np.nanmedian(yerr_fit[np.isfinite(yerr_fit)]))
            yerr_fit = np.where(np.isfinite(yerr_fit), yerr_fit, med)
            yerr_fit = np.nan_to_num(yerr_fit, nan=jitter, posinf=jitter, neginf=jitter)
        else:
            yerr_fit = np.full_like(y_fit, float(jitter), dtype=float)

        y_mean = float(np.mean(y_fit))
        y_fit0 = y_fit - y_mean

        # Build kernel: prefer SHO if S0/w0 provided, otherwise use RealTerm
        if a1 is not None and rho1 is not None and a2 is not None and rho2 is not None:
            # Use RealTerm (OU mixture) if explicitly provided
            k = (
                terms.RealTerm(a=float(a1), c=1.0 / float(rho1)) +
                terms.RealTerm(a=float(a2), c=1.0 / float(rho2))
            )
        else:
            # Default: use SHO kernel with S0, w0, Q
            k = terms.SHOTerm(S0=float(S0), w0=float(w0), Q=float(Q))

        try:
            gp = GaussianProcess(k)
            gp.compute(t_fit, diag=yerr_fit**2)
            mu, var = gp.predict(y_fit0, t, return_var=True)
        except Exception:
            baseline = np.full_like(y, y_med, dtype=float)
            resid = y - baseline
            df_out.loc[idx, "baseline"] = baseline
            df_out.loc[idx, "resid"] = resid
            df_out.loc[idx, "sigma_resid"] = resid / s0
            continue

        baseline = np.asarray(mu, float) + y_mean
        resid = y - baseline

        var = np.asarray(var, float)
        med_err = float(np.nanmedian(yerr_fit)) if np.isfinite(yerr_fit).any() else float(jitter)
        scale = np.sqrt(np.maximum(var, 0.0) + med_err**2)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, max(med_err, 1e-6))
        sigma_resid = resid / scale

        df_out.loc[idx, "baseline"] = baseline
        df_out.loc[idx, "resid"] = resid
        df_out.loc[idx, "sigma_resid"] = sigma_resid

    return df_out
