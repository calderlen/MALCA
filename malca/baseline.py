import numpy as np
import pandas as pd
import warnings

from celerite2 import GaussianProcess, terms

from malca.config.config_pipeline import (
    BASELINE_S0, BASELINE_W0, BASELINE_Q, BASELINE_JITTER,
    GP_FLOOR_CLIP, GP_FLOOR_ITERS, GP_MIN_FLOOR_POINTS,
    GP_AUTO_SCALE_FRACTION, GP_MIN_GP_POINTS,
    GP_DIP_SIGMA_THRESH, GP_PAD_DAYS,
    ROLLING_WINDOW_DAYS, ROLLING_MIN_POINTS, ROLLING_MIN_DAYS,
)


def global_median_baseline(
    df,
    t_col="JD",
    mag_col="mag",
    err_col="error",
    **kwargs,
):
    df_out = df.copy()
    for col in ("baseline", "resid", "sigma_resid", "sigma_eff"):
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
        mad = np.nan

    e_good = np.isfinite(e)
    e_med = float(np.median(e[e_good])) if e_good.any() else np.nan

    mad_num = mad if np.isfinite(mad) else 0.0
    e_med_num = e_med if np.isfinite(e_med) else 0.0
    robust_std = float(np.sqrt(mad_num**2 + e_med_num**2))
    robust_std = max(robust_std, 1e-6)

    sigma_resid = resid / robust_std

    e_safe = np.where(np.isfinite(e) & (e > 0), e, e_med_num)
    sigma_eff = np.sqrt(e_safe**2 + mad_num**2)
    sigma_eff = np.maximum(sigma_eff, 1e-6)

    df_out.loc[:, "baseline"] = baseline
    df_out.loc[:, "resid"] = resid
    df_out.loc[:, "sigma_resid"] = sigma_resid
    df_out.loc[:, "sigma_eff"] = sigma_eff
    return df_out


def rolling_time_median(jd, mag, days=ROLLING_WINDOW_DAYS, min_points=ROLLING_MIN_POINTS, min_days=ROLLING_MIN_DAYS, past_only=True):
    """Rolling time-window median using searchsorted (past-only by default)."""
    n = len(jd)
    out = np.full(n, np.nan, dtype=float)

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

            idx_start = np.searchsorted(jd, lo_val, side="left")
            idx_end = np.searchsorted(jd, hi_val, side="right")
            vals = mag[idx_start:idx_end]
            finite_vals = vals[np.isfinite(vals)]
            if len(finite_vals) >= int(min_points):
                out[i] = np.median(finite_vals)
                break

            window /= 2.0

    return out


def per_camera_median_baseline(
    df,
    days=ROLLING_WINDOW_DAYS,
    min_points=ROLLING_MIN_POINTS,
    t_col="JD",
    mag_col="mag",
    err_col="error",
    cam_col="camera#",
    **kwargs,
):
    df_out = df.copy()
    for col in ("baseline", "resid", "sigma_resid", "sigma_eff"):
        if col not in df_out.columns:
            df_out[col] = np.nan

    for _, sub in df_out.groupby(cam_col, group_keys=False):
        idx = sub.index

        t = df_out.loc[idx, t_col].to_numpy(dtype=float)
        m = df_out.loc[idx, mag_col].to_numpy(dtype=float)
        e = df_out.loc[idx, err_col].to_numpy(dtype=float)

        base = rolling_time_median(t, m, days=days, min_points=min_points)
        resid = m - base

        resid_good = np.isfinite(resid)
        if resid_good.any():
            resid_vals = resid[resid_good]
            mad = float(1.4826 * np.median(np.abs(resid_vals - np.median(resid_vals))))
        else:
            mad = np.nan

        e_good = np.isfinite(e)
        e_med = float(np.median(e[e_good])) if e_good.any() else np.nan

        mad_num = mad if np.isfinite(mad) else 0.0
        e_med_num = e_med if np.isfinite(e_med) else 0.0
        robust_std = float(np.sqrt(mad_num**2 + e_med_num**2))
        robust_std = max(robust_std, 1e-6)

        sigma_resid = resid / robust_std

        e_safe = np.where(np.isfinite(e) & (e > 0), e, e_med_num)
        sigma_eff = np.sqrt(e_safe**2 + mad_num**2)
        sigma_eff = np.maximum(sigma_eff, 1e-6)

        df_out.loc[idx, "baseline"] = base
        df_out.loc[idx, "resid"] = resid
        df_out.loc[idx, "sigma_resid"] = sigma_resid
        df_out.loc[idx, "sigma_eff"] = sigma_eff

    return df_out


def per_camera_gp_baseline(
    df,
    *,
    sigma=None,
    rho=None,
    q=BASELINE_Q,
    S0=BASELINE_S0,
    w0=BASELINE_W0,
    jitter=BASELINE_JITTER,
    t_col="JD",
    mag_col="mag",
    err_col="error",
    cam_col="camera#",
    sigma_floor=None,
    floor_clip=GP_FLOOR_CLIP,
    floor_iters=GP_FLOOR_ITERS,
    min_floor_points=GP_MIN_FLOOR_POINTS,
    add_sigma_eff_col=True,
    auto_scale_gp=True,
    auto_scale_fraction=GP_AUTO_SCALE_FRACTION,
):
    """Per-camera GP baseline (fixed SHO kernel) with sigma_eff output."""
    df_out = df.copy()
    out_cols = ("baseline", "resid", "sigma_resid", "baseline_source") + (("sigma_eff",) if add_sigma_eff_col else ())
    for col in out_cols:
        if col not in df_out.columns:
            df_out[col] = np.nan if col != "baseline_source" else "unknown"

    def robust_sigma_floor(resid, yerr_here, var_here):
        finite0 = np.isfinite(resid) & np.isfinite(yerr_here) & np.isfinite(var_here)
        if finite0.sum() < max(10, min_floor_points):
            return 0.0

        r = resid[finite0].copy()
        keep = np.ones_like(r, dtype=bool)
        for _ in range(int(max(floor_iters, 1))):
            rr = r[keep]
            if rr.size < max(10, min_floor_points):
                break
            med = float(np.median(rr))
            mad = 1.4826 * float(np.median(np.abs(rr - med)))
            mad = max(mad, 1e-12)
            keep = np.abs(r - med) <= float(floor_clip) * mad

        rr = r[keep]
        if rr.size < max(10, min_floor_points):
            rr = r

        s_quiet = 1.4826 * float(np.median(np.abs(rr - float(np.median(rr)))))
        s_quiet = max(s_quiet, 1e-12)

        yerr2_med = float(
            np.median(
                (yerr_here[finite0][keep] if keep.size == yerr_here[finite0].size else yerr_here[finite0]) ** 2
            )
        )
        var_med = float(
            np.median((var_here[finite0][keep] if keep.size == var_here[finite0].size else var_here[finite0]))
        )

        floor2 = max(s_quiet**2 - yerr2_med - var_med, 0.0)
        return float(np.sqrt(floor2))

    for _, sub in df_out.groupby(cam_col, group_keys=False):
        idx = sub.sort_values(t_col).index
        t = df_out.loc[idx, t_col].to_numpy(dtype=float)
        y = df_out.loc[idx, mag_col].to_numpy(dtype=float)
        yerr = df_out.loc[idx, err_col].to_numpy(dtype=float)

        finite = np.isfinite(t) & np.isfinite(y)
        if finite.sum() < 5:
            if np.isfinite(y).any():
                baseline_val = float(np.nanmedian(y[np.isfinite(y)]))
                baseline = np.full_like(y, baseline_val, dtype=float)
                resid = y - baseline

                if np.isfinite(yerr).any():
                    med_yerr_all = float(np.nanmedian(yerr[np.isfinite(yerr)]))
                else:
                    med_yerr_all = float(jitter)
                yerr_full = np.where(np.isfinite(yerr), yerr, med_yerr_all)
                yerr_full = np.nan_to_num(yerr_full, nan=float(jitter), posinf=float(jitter), neginf=float(jitter))
                yerr_full = np.maximum(yerr_full, 0.0)

                sigma_eff = np.sqrt(yerr_full**2 + float(jitter) ** 2)
                sigma_resid = resid / sigma_eff

                df_out.loc[idx, "baseline"] = baseline
                df_out.loc[idx, "resid"] = resid
                df_out.loc[idx, "sigma_resid"] = sigma_resid
                if add_sigma_eff_col:
                    df_out.loc[idx, "sigma_eff"] = sigma_eff
                df_out.loc[idx, "baseline_source"] = "median_fallback"
            continue

        finite_idx = np.flatnonzero(finite)
        t_fit = t[finite_idx]
        y_fit = y[finite_idx]
        y_mean = float(np.mean(y_fit))
        y_centered = y_fit - y_mean

        yerr_fit = yerr[finite_idx]
        if not np.isfinite(yerr_fit).any():
            yerr_fit = np.full_like(y_fit, float(jitter), dtype=float)
        else:
            med_yerr = float(np.nanmedian(yerr_fit[np.isfinite(yerr_fit)]))
            med_yerr = float(med_yerr) if np.isfinite(med_yerr) else float(jitter)
            yerr_fit = np.where(np.isfinite(yerr_fit), yerr_fit, med_yerr)
            yerr_fit = np.nan_to_num(yerr_fit, nan=float(jitter), posinf=float(jitter), neginf=float(jitter))

        if sigma is not None and rho is not None:
            k = terms.SHOTerm(sigma=float(sigma), rho=float(rho), Q=float(q))
        else:
            w0_use = float(w0)
            if auto_scale_gp:
                time_span = float(t_fit.max() - t_fit.min())
                if time_span > 0:
                    target_timescale = max(time_span * float(auto_scale_fraction), 50.0)
                    w0_scaled = 2.0 * np.pi / target_timescale
                    w0_use = max(w0_scaled, float(w0))
            k = terms.SHOTerm(S0=float(S0), w0=w0_use, Q=float(q))

        baseline = np.full_like(y, np.nan, dtype=float)
        var = np.zeros_like(y, dtype=float)
        baseline_flag = "median_fallback"

        try:
            gp = GaussianProcess(k)
            gp.compute(t_fit, diag=yerr_fit**2)
            mu, var_pred = gp.predict(y_centered, t, return_var=True)
            baseline = np.asarray(mu, dtype=float) + y_mean
            var = np.asarray(var_pred, dtype=float)
            var = np.where(np.isfinite(var) & (var >= 0.0), var, 0.0)
            baseline_flag = "gp_sho"
        except Exception as exc:
            warnings.warn(f"GP fit failed for camera group; falling back to median baseline. Error: {exc}")
            y_med = float(np.nanmedian(y[finite]))
            baseline = np.full_like(y, y_med, dtype=float)
            var = np.zeros_like(y, dtype=float)
            baseline_flag = "median_fallback"

        resid = y - baseline

        if np.isfinite(yerr).any():
            med_yerr_all = float(np.nanmedian(yerr[np.isfinite(yerr)]))
            med_yerr_all = med_yerr_all if np.isfinite(med_yerr_all) else float(jitter)
        else:
            med_yerr_all = float(jitter)
        yerr_full = np.where(np.isfinite(yerr), yerr, med_yerr_all)
        yerr_full = np.nan_to_num(yerr_full, nan=float(jitter), posinf=float(jitter), neginf=float(jitter))
        yerr_full = np.maximum(yerr_full, 0.0)

        if sigma_floor is None:
            floor_here = robust_sigma_floor(resid, yerr_full, var)
        else:
            floor_here = float(max(sigma_floor, 0.0))

        sigma_eff2 = yerr_full**2 + floor_here**2 + var
        sigma_eff = np.sqrt(np.maximum(sigma_eff2, 1e-12))
        sigma_resid = resid / sigma_eff

        df_out.loc[idx, "baseline"] = baseline
        df_out.loc[idx, "resid"] = resid
        df_out.loc[idx, "sigma_resid"] = sigma_resid
        if add_sigma_eff_col:
            df_out.loc[idx, "sigma_eff"] = sigma_eff
        df_out.loc[idx, "baseline_source"] = baseline_flag

    return df_out


def per_camera_gp_baseline_masked(
    df,
    *,
    dip_sigma_thresh=GP_DIP_SIGMA_THRESH,
    pad_days=GP_PAD_DAYS,
    S0=BASELINE_S0,
    w0=BASELINE_W0,
    q=BASELINE_Q,
    a1=None,
    rho1=None,
    a2=None,
    rho2=None,
    jitter=BASELINE_JITTER,
    use_yerr=True,
    t_col="JD",
    mag_col="mag",
    err_col="error",
    cam_col="camera#",
    min_gp_points=GP_MIN_GP_POINTS,
    add_sigma_eff_col=True,
    sigma_floor=None,
    floor_clip=GP_FLOOR_CLIP,
    floor_iters=GP_FLOOR_ITERS,
    min_floor_points=GP_MIN_FLOOR_POINTS,
    auto_scale_gp=True,
    auto_scale_fraction=GP_AUTO_SCALE_FRACTION,
    **kwargs,
):
    """Per-camera GP baseline with dip masking (excludes significant dips from fit)."""

    def robust_sigma_floor(resid, yerr_here, var_here):
        finite0 = np.isfinite(resid) & np.isfinite(yerr_here) & np.isfinite(var_here)
        if finite0.sum() < max(10, min_floor_points):
            return 0.0

        r = resid[finite0].copy()
        keep = np.ones_like(r, dtype=bool)
        for _ in range(int(max(floor_iters, 1))):
            rr = r[keep]
            if rr.size < max(10, min_floor_points):
                break
            med = float(np.median(rr))
            mad = 1.4826 * float(np.median(np.abs(rr - med)))
            mad = max(mad, 1e-12)
            keep = np.abs(r - med) <= float(floor_clip) * mad

        rr = r[keep]
        if rr.size < max(10, min_floor_points):
            rr = r

        s_quiet = 1.4826 * float(np.median(np.abs(rr - float(np.median(rr)))))
        s_quiet = max(s_quiet, 1e-12)

        yerr2_med = float(
            np.median(
                (yerr_here[finite0][keep] if keep.size == yerr_here[finite0].size else yerr_here[finite0]) ** 2
            )
        )
        var_med = float(
            np.median((var_here[finite0][keep] if keep.size == var_here[finite0].size else var_here[finite0]))
        )
        floor2 = max(s_quiet**2 - yerr2_med - var_med, 0.0)
        return float(np.sqrt(floor2))

    df_out = df.copy()
    out_cols = ("baseline", "resid", "sigma_resid") + (("sigma_eff",) if add_sigma_eff_col else ())
    for col in out_cols:
        if col not in df_out.columns:
            df_out[col] = np.nan

    for _, sub in df_out.groupby(cam_col, group_keys=False):
        idx = sub.sort_values(t_col).index
        t = df_out.loc[idx, t_col].to_numpy(float)
        y = df_out.loc[idx, mag_col].to_numpy(float)

        if use_yerr:
            yerr = df_out.loc[idx, err_col].to_numpy(float)
        else:
            yerr = np.full_like(y, np.nan, dtype=float)

        finite = np.isfinite(t) & np.isfinite(y)
        y_med = float(np.nanmedian(y[finite]))
        r0 = y - y_med

        r0_f = r0[finite]
        med_r = float(np.nanmedian(r0_f))
        mad_r = 1.4826 * float(np.nanmedian(np.abs(r0_f - med_r)))

        if use_yerr and np.isfinite(yerr).any():
            e_med = float(np.nanmedian(yerr[finite & np.isfinite(yerr)]))
        else:
            e_med = float(jitter)

        s0 = float(np.sqrt(max(mad_r, 0.0) ** 2 + max(e_med, 0.0) ** 2))
        s0 = max(s0, 1e-6)

        sig0 = r0 / s0
        dip_flag = finite & np.isfinite(sig0) & (sig0 < float(dip_sigma_thresh))

        keep = finite.copy()
        if dip_flag.any():
            t_dip = t[dip_flag]
            bad = np.zeros_like(keep, dtype=bool)
            for td in t_dip:
                bad |= np.abs(t - td) <= float(pad_days)
            keep &= ~bad

        if keep.sum() < min_gp_points:
            baseline = np.full_like(y, y_med, dtype=float)
            resid = y - baseline

            if use_yerr and np.isfinite(yerr).any():
                yerr_full = np.where(np.isfinite(yerr), yerr, e_med)
            else:
                yerr_full = np.full_like(y, e_med, dtype=float)

            floor_here = float(max(sigma_floor, 0.0)) if sigma_floor is not None else float(jitter)
            sigma_eff = np.sqrt(yerr_full**2 + floor_here**2)
            sigma_resid = resid / sigma_eff

            df_out.loc[idx, "baseline"] = baseline
            df_out.loc[idx, "resid"] = resid
            df_out.loc[idx, "sigma_resid"] = sigma_resid
            if add_sigma_eff_col:
                df_out.loc[idx, "sigma_eff"] = sigma_eff
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

        if a1 is not None and rho1 is not None and a2 is not None and rho2 is not None:
            k = terms.RealTerm(a=float(a1), c=1.0 / float(rho1)) + terms.RealTerm(a=float(a2), c=1.0 / float(rho2))
        else:
            w0_use = float(w0)
            if auto_scale_gp:
                time_span = float(t_fit.max() - t_fit.min())
                if time_span > 0:
                    target_timescale = max(time_span * float(auto_scale_fraction), 50.0)
                    w0_scaled = 2.0 * np.pi / target_timescale
                    w0_use = max(w0_scaled, float(w0))
            k = terms.SHOTerm(S0=float(S0), w0=w0_use, Q=float(q))

        try:
            gp = GaussianProcess(k)
            gp.compute(t_fit, diag=yerr_fit**2)
            mu, var = gp.predict(y_fit0, t, return_var=True)
        except Exception:
            baseline = np.full_like(y, y_med, dtype=float)
            resid = y - baseline

            if use_yerr and np.isfinite(yerr).any():
                yerr_full = np.where(np.isfinite(yerr), yerr, e_med)
            else:
                yerr_full = np.full_like(y, e_med, dtype=float)

            floor_here = float(max(sigma_floor, 0.0)) if sigma_floor is not None else float(jitter)
            sigma_eff = np.sqrt(yerr_full**2 + floor_here**2)
            sigma_resid = resid / sigma_eff

            df_out.loc[idx, "baseline"] = baseline
            df_out.loc[idx, "resid"] = resid
            df_out.loc[idx, "sigma_resid"] = sigma_resid
            if add_sigma_eff_col:
                df_out.loc[idx, "sigma_eff"] = sigma_eff
            continue

        baseline = np.asarray(mu, float) + y_mean
        resid = y - baseline

        var = np.asarray(var, float)
        var = np.maximum(var, 0.0)

        if use_yerr and np.isfinite(yerr).any():
            yerr_full = np.where(np.isfinite(yerr), yerr, e_med)
        else:
            yerr_full = np.full_like(y, e_med, dtype=float)

        if sigma_floor is None:
            floor_here = robust_sigma_floor(resid, yerr_full, var)
        else:
            floor_here = float(max(sigma_floor, 0.0))

        sigma_eff2 = yerr_full**2 + floor_here**2 + var
        sigma_eff = np.sqrt(np.maximum(sigma_eff2, 1e-12))
        sigma_resid = resid / sigma_eff

        if add_sigma_eff_col:
            df_out.loc[idx, "sigma_eff"] = sigma_eff

        df_out.loc[idx, "baseline"] = baseline
        df_out.loc[idx, "resid"] = resid
        df_out.loc[idx, "sigma_resid"] = sigma_resid

    return df_out

