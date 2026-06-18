import warnings

from celerite2 import GaussianProcess, terms
import numpy as np
import pandas as pd

from malca.config import (
    PHASE_TEMPLATE_MIN_POINTS,
    PHASE_TEMPLATE_PHASE_BINS,
    PHASE_TEMPLATE_PROFILE_SMOOTH_WINDOW,
)
from malca.config import (
    BASELINE_S0, BASELINE_W0, BASELINE_Q, BASELINE_JITTER,
    GP_FLOOR_CLIP, GP_FLOOR_ITERS, GP_MIN_FLOOR_POINTS,
    GP_STIFF_SCALE_FRACTION, GP_STIFF_MIN_DAYS,
    GP_LOOSE_SCALE_FRACTION, GP_LOOSE_MIN_DAYS, GP_MIN_GP_POINTS,
    GP_DIP_SIGMA_THRESH, GP_BRIGHT_SIGMA_THRESH, GP_PAD_DAYS,
    GP_LATE_ONSET_BUFFER_DAYS, GP_MIN_ANCHOR_OVERLAP_DAYS,
    GP_MIN_QUIET_BASELINE_DAYS, GP_CONSENSUS_START_RESID_SIGMA,
    ROLLING_WINDOW_DAYS, ROLLING_MIN_POINTS, ROLLING_MIN_DAYS,
)
from malca.config import MAD_SCALE



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
        mad = float(MAD_SCALE * np.median(np.abs(resid_vals - med_resid)))
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
    df_out.loc[:, "baseline_source"] = "global_median"
    return df_out


def rolling_time_median(jd, mag, days=ROLLING_WINDOW_DAYS, min_points=ROLLING_MIN_POINTS, min_days=ROLLING_MIN_DAYS, past_only=False):
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
            mad = float(MAD_SCALE * np.median(np.abs(resid_vals - np.median(resid_vals))))
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
        df_out.loc[idx, "baseline_source"] = "per_camera_median"

    return df_out


def _robust_sigma(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        return np.nan

    med = float(np.median(vals))
    mad = float(MAD_SCALE * np.median(np.abs(vals - med)))
    if np.isfinite(mad) and mad > 0:
        return mad

    sigma = float(np.nanstd(vals))
    return sigma if np.isfinite(sigma) and sigma > 0 else np.nan


def _circular_fill_and_smooth_template(
    template: np.ndarray,
    *,
    smooth_window: int,
) -> np.ndarray | None:
    values = np.asarray(template, dtype=float)
    n_bins = int(values.size)
    finite = np.isfinite(values)
    if n_bins == 0 or not finite.any():
        return None

    if finite.sum() == 1:
        filled = np.full(n_bins, float(values[finite][0]), dtype=float)
    else:
        centers = (np.arange(n_bins, dtype=float) + 0.5) / float(n_bins)
        xp = np.concatenate([centers[finite] - 1.0, centers[finite], centers[finite] + 1.0])
        fp = np.concatenate([values[finite], values[finite], values[finite]])
        filled = np.interp(centers, xp, fp)

    window = max(int(smooth_window), 1)
    if window % 2 == 0:
        window += 1
    if window <= 1:
        smoothed = filled
    else:
        pad = window // 2
        kernel = np.ones(window, dtype=float) / float(window)
        ext = np.concatenate([filled[-pad:], filled, filled[:pad]])
        smoothed = np.convolve(ext, kernel, mode="valid")

    finite_smooth = np.isfinite(smoothed)
    if finite_smooth.any():
        smoothed = smoothed - float(np.median(smoothed[finite_smooth]))
    return smoothed


def _build_phase_template_model(
    jd: np.ndarray,
    centered_mag: np.ndarray,
    *,
    period_days: float,
    phase_bins: int,
    smooth_window: int,
    min_bin_points: int,
) -> np.ndarray | None:
    valid = np.isfinite(jd) & np.isfinite(centered_mag)
    if np.count_nonzero(valid) == 0:
        return None

    jd_valid = np.asarray(jd[valid], dtype=float)
    centered_valid = np.asarray(centered_mag[valid], dtype=float)
    jd0 = float(np.nanmin(jd_valid))
    phase_valid = np.mod((jd_valid - jd0) / float(period_days), 1.0)

    n_bins = max(int(phase_bins), 4)
    template = np.full(n_bins, np.nan, dtype=float)
    bin_idx = np.floor(phase_valid * n_bins).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    for b in range(n_bins):
        vals = centered_valid[bin_idx == b]
        if vals.size >= int(min_bin_points):
            template[b] = float(np.median(vals))

    if np.count_nonzero(np.isfinite(template)) < max(6, n_bins // 4):
        return None

    smoothed = _circular_fill_and_smooth_template(template, smooth_window=smooth_window)
    if smoothed is None:
        return None

    centers = (np.arange(n_bins, dtype=float) + 0.5) / float(n_bins)
    xp = np.concatenate([centers - 1.0, centers, centers + 1.0])
    fp = np.concatenate([smoothed, smoothed, smoothed])
    phase_all = np.mod((np.asarray(jd, dtype=float) - jd0) / float(period_days), 1.0)
    return np.interp(phase_all, xp, fp)


def _phase_template_offsets(
    df: pd.DataFrame,
    *,
    mag_col: str,
    cam_col: str,
    band_col: str,
    min_camera_band_points: int,
) -> np.ndarray:
    mag = pd.to_numeric(df[mag_col], errors="coerce")
    finite_mag = mag[np.isfinite(mag)]
    global_median = float(np.median(finite_mag)) if not finite_mag.empty else 0.0

    offsets = np.full(len(df), global_median, dtype=float)
    work = pd.DataFrame(index=df.index)
    work["_mag"] = mag

    if cam_col in df.columns:
        work["_camera"] = df[cam_col]
        camera_median = work.groupby("_camera")["_mag"].median()
        work = work.join(camera_median.rename("_camera_median"), on="_camera")
        camera_vals = work["_camera_median"].to_numpy(dtype=float)
        camera_mask = np.isfinite(camera_vals)
        offsets[camera_mask] = camera_vals[camera_mask]

    if band_col in df.columns:
        work["_band"] = pd.to_numeric(df[band_col], errors="coerce")
        band_median = work.groupby("_band")["_mag"].median()
        work = work.join(band_median.rename("_band_median"), on="_band")
        band_vals = work["_band_median"].to_numpy(dtype=float)
        band_mask = np.isfinite(band_vals)
        offsets[band_mask] = band_vals[band_mask]

        if "_camera" in work.columns:
            camera_band = (
                work.groupby(["_band", "_camera"], dropna=False)["_mag"]
                .agg(["median", "size"])
                .rename(columns={"median": "_camera_band_median", "size": "_camera_band_size"})
            )
            work = work.join(camera_band, on=["_band", "_camera"])
            camera_band_size = pd.to_numeric(work["_camera_band_size"], errors="coerce").to_numpy(dtype=float)
            camera_band_median = pd.to_numeric(work["_camera_band_median"], errors="coerce").to_numpy(dtype=float)
            use_camera_band = (
                np.isfinite(camera_band_size)
                & (camera_band_size >= int(min_camera_band_points))
                & np.isfinite(camera_band_median)
            )
            offsets[use_camera_band] = camera_band_median[use_camera_band]

    return offsets


def phase_template_baseline(
    df,
    *,
    period_days=None,
    phase_bins=PHASE_TEMPLATE_PHASE_BINS,
    profile_smooth_window=PHASE_TEMPLATE_PROFILE_SMOOTH_WINDOW,
    min_points=PHASE_TEMPLATE_MIN_POINTS,
    min_bin_points=3,
    min_camera_band_points=8,
    t_col="JD",
    mag_col="mag",
    err_col="error",
    cam_col="camera#",
    band_col="v_g_band",
    **kwargs,
):
    try:
        period_value = float(period_days)
    except (TypeError, ValueError):
        period_value = np.nan

    def _fallback() -> pd.DataFrame:
        if cam_col in df.columns:
            out = per_camera_median_baseline(
                df,
                t_col=t_col,
                mag_col=mag_col,
                err_col=err_col,
                cam_col=cam_col,
            )
        else:
            out = global_median_baseline(
                df,
                t_col=t_col,
                mag_col=mag_col,
                err_col=err_col,
            )
        out.loc[:, "baseline_source"] = "phase_template_fallback"
        return out

    if not np.isfinite(period_value) or period_value <= 0:
        return _fallback()

    df_out = df.copy()
    for col in ("baseline", "resid", "sigma_resid", "sigma_eff", "baseline_source"):
        if col not in df_out.columns:
            df_out[col] = np.nan if col != "baseline_source" else "unknown"

    jd = pd.to_numeric(df_out[t_col], errors="coerce").to_numpy(dtype=float)
    mag = pd.to_numeric(df_out[mag_col], errors="coerce").to_numpy(dtype=float)
    err = pd.to_numeric(df_out[err_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(jd) & np.isfinite(mag)
    if np.count_nonzero(finite) < int(min_points):
        return _fallback()

    offsets = _phase_template_offsets(
        df_out,
        mag_col=mag_col,
        cam_col=cam_col,
        band_col=band_col,
        min_camera_band_points=min_camera_band_points,
    )
    centered_mag = mag - offsets
    template_model = _build_phase_template_model(
        jd,
        centered_mag,
        period_days=period_value,
        phase_bins=int(phase_bins),
        smooth_window=int(profile_smooth_window),
        min_bin_points=int(min_bin_points),
    )
    if template_model is None or not np.isfinite(template_model).any():
        return _fallback()

    baseline = template_model + offsets
    resid = mag - baseline
    df_out.loc[:, "baseline"] = baseline
    df_out.loc[:, "resid"] = resid
    df_out.loc[:, "baseline_source"] = "phase_template"

    if cam_col in df_out.columns:
        grouped = df_out.groupby(cam_col, group_keys=False)
    else:
        grouped = [(None, df_out)]

    for _, sub in grouped:
        idx = sub.index
        resid_here = df_out.loc[idx, "resid"].to_numpy(dtype=float)
        err_here = pd.to_numeric(df_out.loc[idx, err_col], errors="coerce").to_numpy(dtype=float)
        scatter = _robust_sigma(resid_here)
        err_finite = err_here[np.isfinite(err_here) & (err_here > 0)]
        err_med = float(np.median(err_finite)) if err_finite.size else np.nan

        scatter_num = scatter if np.isfinite(scatter) else 0.0
        err_med_num = err_med if np.isfinite(err_med) else 0.0
        robust_std = float(np.sqrt(scatter_num**2 + err_med_num**2))
        robust_std = max(robust_std, 1e-6)

        sigma_resid = resid_here / robust_std
        err_safe = np.where(np.isfinite(err_here) & (err_here > 0), err_here, err_med_num)
        sigma_eff = np.sqrt(err_safe**2 + scatter_num**2)
        sigma_eff = np.maximum(sigma_eff, 1e-6)

        df_out.loc[idx, "sigma_resid"] = sigma_resid
        df_out.loc[idx, "sigma_eff"] = sigma_eff

    if not np.isfinite(pd.to_numeric(df_out["sigma_eff"], errors="coerce")).any():
        return _fallback()

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
    mag_err_col="error",
    cam_col="camera#",
    sigma_floor=None,
    floor_clip=GP_FLOOR_CLIP,
    floor_iters=GP_FLOOR_ITERS,
    min_floor_points=GP_MIN_FLOOR_POINTS,
    add_sigma_eff_col=True,
    auto_scale_gp=True,
    loose_scale_fraction=GP_LOOSE_SCALE_FRACTION,
    loose_min_days=GP_LOOSE_MIN_DAYS,
):
    """Per-camera GP baseline (fixed SHO kernel) with sigma_eff output."""
    df_out = df.copy()
    out_cols = ("baseline", "resid", "sigma_resid", "baseline_source") + (("sigma_eff",) if add_sigma_eff_col else ())
    for col in out_cols:
        if col not in df_out.columns:
            df_out[col] = np.nan if col != "baseline_source" else "unknown"

    def robust_sigma_floor(resid, mag_err_here, var_here):
        finite0 = np.isfinite(resid) & np.isfinite(mag_err_here) & np.isfinite(var_here)
        if finite0.sum() < max(10, min_floor_points):
            return 0.0

        r = resid[finite0].copy()
        keep = np.ones_like(r, dtype=bool)
        for _ in range(int(max(floor_iters, 1))):
            rr = r[keep]
            if rr.size < max(10, min_floor_points):
                break
            med = float(np.median(rr))
            mad = MAD_SCALE * float(np.median(np.abs(rr - med)))
            mad = max(mad, 1e-12)
            keep = np.abs(r - med) <= float(floor_clip) * mad

        rr = r[keep]
        if rr.size < max(10, min_floor_points):
            rr = r

        s_quiet = MAD_SCALE * float(np.median(np.abs(rr - float(np.median(rr)))))
        s_quiet = max(s_quiet, 1e-12)

        mag_err2_med = float(
            np.median(
                (mag_err_here[finite0][keep] if keep.size == mag_err_here[finite0].size else mag_err_here[finite0]) ** 2
            )
        )
        var_med = float(
            np.median((var_here[finite0][keep] if keep.size == var_here[finite0].size else var_here[finite0]))
        )

        floor2 = max(s_quiet**2 - mag_err2_med - var_med, 0.0)
        return float(np.sqrt(floor2))

    for _, sub in df_out.groupby(cam_col, group_keys=False):
        idx = sub.sort_values(t_col).index
        t = df_out.loc[idx, t_col].to_numpy(dtype=float)
        mag = df_out.loc[idx, mag_col].to_numpy(dtype=float)
        mag_err = df_out.loc[idx, mag_err_col].to_numpy(dtype=float)

        finite = np.isfinite(t) & np.isfinite(mag)
        if finite.sum() < 5:
            if np.isfinite(mag).any():
                baseline_val = float(np.nanmedian(mag[np.isfinite(mag)]))
                baseline = np.full_like(mag, baseline_val, dtype=float)
                resid = mag - baseline

                sigma_eff = np.sqrt(mag_err**2 + float(jitter) ** 2)
                sigma_resid = resid / sigma_eff

                df_out.loc[idx, "baseline"] = baseline
                df_out.loc[idx, "resid"] = resid
                df_out.loc[idx, "sigma_resid"] = sigma_resid
                if add_sigma_eff_col:
                    df_out.loc[idx, "sigma_eff"] = sigma_eff
                df_out.loc[idx, "baseline_source"] = "median_fallback"
            continue

        fit_finite = finite & np.isfinite(mag_err) & (mag_err > 0)
        finite_idx = np.flatnonzero(fit_finite)
        if finite_idx.size < 5:
            baseline_val = float(np.nanmedian(mag[finite]))
            baseline = np.full_like(mag, baseline_val, dtype=float)
            resid = mag - baseline
            sigma_eff = np.sqrt(mag_err**2 + float(jitter) ** 2)
            sigma_resid = resid / sigma_eff

            df_out.loc[idx, "baseline"] = baseline
            df_out.loc[idx, "resid"] = resid
            df_out.loc[idx, "sigma_resid"] = sigma_resid
            if add_sigma_eff_col:
                df_out.loc[idx, "sigma_eff"] = sigma_eff
            df_out.loc[idx, "baseline_source"] = "median_fallback"
            continue

        t_fit = t[finite_idx]
        mag_fit = mag[finite_idx]
        mean_mag = float(np.mean(mag_fit))
        mag_centered = mag_fit - mean_mag

        mag_err_fit = mag_err[finite_idx]

        if sigma is not None and rho is not None:
            k = terms.SHOTerm(sigma=float(sigma), rho=float(rho), Q=float(q))
        else:
            w0_use = float(w0)
            if auto_scale_gp:
                time_span = float(t_fit.max() - t_fit.min())
                if time_span > 0:
                    target_timescale = max(time_span * float(loose_scale_fraction), float(loose_min_days))
                    w0_scaled = 2.0 * np.pi / target_timescale
                    w0_use = max(w0_scaled, float(w0))
            k = terms.SHOTerm(S0=float(S0), w0=w0_use, Q=float(q))

        baseline = np.full_like(mag, np.nan, dtype=float)
        var = np.zeros_like(mag, dtype=float)
        baseline_flag = "median_fallback"

        try:
            gp = GaussianProcess(k)
            gp.compute(t_fit, diag=mag_err_fit**2)
            mean_prediction, var_pred = gp.predict(mag_centered, t, return_var=True)
            baseline = np.asarray(mean_prediction, dtype=float) + mean_mag
            var = np.asarray(var_pred, dtype=float)
            var = np.where(np.isfinite(var) & (var >= 0.0), var, 0.0)
            baseline_flag = "gp_sho"
        except Exception as exc:
            warnings.warn(f"GP fit failed for camera group; falling back to median baseline. Error: {exc}")
            median_mag = float(np.nanmedian(mag[finite]))
            baseline = np.full_like(mag, median_mag, dtype=float)
            var = np.zeros_like(mag, dtype=float)
            baseline_flag = "median_fallback"

        resid = mag - baseline

        mag_err_full = mag_err

        if sigma_floor is None:
            floor_here = robust_sigma_floor(resid, mag_err_full, var)
        else:
            floor_here = float(max(sigma_floor, 0.0))

        sigma_eff2 = mag_err_full**2 + floor_here**2 + var
        sigma_eff = np.sqrt(np.maximum(sigma_eff2, 1e-12))
        sigma_resid = resid / sigma_eff

        df_out.loc[idx, "baseline"] = baseline
        df_out.loc[idx, "resid"] = resid
        df_out.loc[idx, "sigma_resid"] = sigma_resid
        if add_sigma_eff_col:
            df_out.loc[idx, "sigma_eff"] = sigma_eff
        df_out.loc[idx, "baseline_source"] = baseline_flag

    return df_out


def _fit_stiff_gp(
    t,
    mag,
    mag_err,
    finite,
    *,
    auto_scale_gp,
    min_gp_points,
    stiff_scale_fraction,
    stiff_min_days,
    S0,
    q,
):
    """Fit a stiff SHO GP for masking; returns base_rough or None."""
    if not (auto_scale_gp and finite.sum() >= min_gp_points):
        return None
    try:
        fit_finite = finite & np.isfinite(mag_err) & (mag_err > 0)
        if np.count_nonzero(fit_finite) < min_gp_points:
            raise ValueError("insufficient finite mag_err points for stiff GP")

        t_stiff = t[fit_finite]
        mag_stiff = mag[fit_finite]
        mean_mag_stiff = float(np.mean(mag_stiff))
        mag_stiff_centered = mag_stiff - mean_mag_stiff
        mag_err_stiff = mag_err[fit_finite]

        time_span_stiff = float(t_stiff.max() - t_stiff.min())
        target_timescale_stiff = max(
            time_span_stiff * float(stiff_scale_fraction), float(stiff_min_days)
        )
        w0_stiff = 2.0 * np.pi / target_timescale_stiff
        k_stiff = terms.SHOTerm(S0=float(S0), w0=w0_stiff, Q=float(q))

        gp_stiff = GaussianProcess(k_stiff)
        gp_stiff.compute(t_stiff, diag=mag_err_stiff**2)
        mean_prediction_stiff = gp_stiff.predict(
            mag_stiff_centered, t, return_var=False
        )

        return np.asarray(mean_prediction_stiff, dtype=float) + mean_mag_stiff
    except Exception:
        return None


def _compute_base_rough(
    t,
    mag,
    mag_err,
    finite,
    median_mag,
    *,
    auto_scale_gp,
    min_gp_points,
    stiff_scale_fraction,
    stiff_min_days,
    S0,
    q,
):
    base_rough = _fit_stiff_gp(
        t,
        mag,
        mag_err,
        finite,
        auto_scale_gp=auto_scale_gp,
        min_gp_points=min_gp_points,
        stiff_scale_fraction=stiff_scale_fraction,
        stiff_min_days=stiff_min_days,
        S0=S0,
        q=q,
    )
    if base_rough is None:
        base_rough = rolling_time_median(t, mag, past_only=False)
    return np.where(np.isfinite(base_rough), base_rough, median_mag)


def _mask_excursion_sigma(
    mag,
    base_rough,
    finite,
    mag_err,
):
    rough_resid = mag - base_rough
    rough_resid_finite = rough_resid[finite]
    median_rough_resid = float(np.nanmedian(rough_resid_finite))
    mad_rough_resid = MAD_SCALE * float(
        np.nanmedian(np.abs(rough_resid_finite - median_rough_resid))
    )
    median_mag_err = float(np.nanmedian(mag_err[finite & np.isfinite(mag_err)]))
    s0 = float(np.sqrt(max(mad_rough_resid, 0.0) ** 2 + max(median_mag_err, 0.0) ** 2))
    return max(s0, 1e-6)


def _mask_excursions(
    t,
    mag,
    base_rough,
    finite,
    median_mag,
    *,
    mag_err,
    dip_sigma_thresh,
    bright_sigma_thresh,
    pad_days,
):
    """Stateful dip/flare masking; returns flags, keep mask, and padded intervals."""
    s0 = _mask_excursion_sigma(mag, base_rough, finite, mag_err)

    flags = np.zeros(len(t), dtype=bool)
    in_dip = False
    in_flare = False

    valid_base_idx = np.where(np.isfinite(base_rough) & finite)[0]
    ref_baseline = base_rough[valid_base_idx[0]] if len(valid_base_idx) > 0 else median_mag

    thresh_dip = float(dip_sigma_thresh)
    thresh_bright = float(bright_sigma_thresh)

    for i in range(len(t)):
        if not finite[i]:
            continue

        if not (in_dip or in_flare):
            ref_baseline = base_rough[i]

        sig = (mag[i] - ref_baseline) / s0

        if sig > thresh_dip:
            in_dip = True
            in_flare = False
            flags[i] = True
        elif in_dip and sig > 1.0:
            flags[i] = True
        else:
            in_dip = False

        if not in_dip:
            if sig < thresh_bright:
                in_flare = True
                flags[i] = True
            elif in_flare and sig < -1.0:
                flags[i] = True
            else:
                in_flare = False

    event_flag = finite & flags
    keep = finite.copy()
    intervals: list[tuple[float, float]] = []
    if event_flag.any():
        t_event = t[event_flag]
        bad = np.zeros_like(keep, dtype=bool)
        for td in t_event:
            lo = float(td) - float(pad_days)
            hi = float(td) + float(pad_days)
            bad |= (t >= lo) & (t <= hi)
            intervals.append((lo, hi))
        keep &= ~bad

    return flags, keep, _union_intervals(intervals), s0


def _union_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_iv[0]]
    for lo, hi in sorted_iv[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo <= prev_hi:
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _time_in_intervals(t_val: float, intervals: list[tuple[float, float]]) -> bool:
    for lo, hi in intervals:
        if lo <= float(t_val) <= hi:
            return True
    return False


def _quiet_span_before_first_excursion(
    t_first: float,
    t_last: float,
    intervals: list[tuple[float, float]],
) -> float:
    """Days from first observation to the start of the first overlapping excursion."""
    overlapping = [
        (lo, hi) for lo, hi in intervals if hi >= float(t_first) and lo <= float(t_last)
    ]
    if not overlapping:
        return float(t_last) - float(t_first)
    first_start = min(lo for lo, _hi in overlapping)
    return max(float(first_start) - float(t_first), 0.0)


def _camera_overlap_days(
    t_a_min: float,
    t_a_max: float,
    t_b_min: float,
    t_b_max: float,
) -> float:
    return max(0.0, min(float(t_a_max), float(t_b_max)) - max(float(t_a_min), float(t_b_min)))


def _build_band_consensus(
    anchor_base_rough,
    band_id,
    t_target,
    anchor_ids,
    cam_band_map,
    *,
    min_anchor_overlap_days,
):
    """Robust median of selected anchor base_rough curves at t_target epochs."""
    contributions = []
    t_min = float(np.min(t_target))
    t_max = float(np.max(t_target))
    target_span = t_max - t_min
    point_target = target_span < float(min_anchor_overlap_days)
    for cam_id in anchor_ids:
        if cam_id not in anchor_base_rough:
            continue
        if cam_band_map.get(cam_id) != band_id:
            continue
        t_anchor, br_anchor = anchor_base_rough[cam_id]
        anchor_min = float(t_anchor.min())
        anchor_max = float(t_anchor.max())
        if point_target:
            in_range = (t_target >= anchor_min) & (t_target <= anchor_max)
            if not np.any(in_range):
                continue
            interped = np.interp(t_target, t_anchor, br_anchor)
            interped[~in_range] = np.nan
        else:
            overlap = _camera_overlap_days(t_min, t_max, anchor_min, anchor_max)
            if overlap < float(min_anchor_overlap_days):
                continue
            interped = np.interp(t_target, t_anchor, br_anchor)
            in_range = (t_target >= anchor_min) & (t_target <= anchor_max)
            interped[~in_range] = np.nan
        contributions.append(interped)

    if not contributions:
        return None
    stack = np.column_stack(contributions)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        consensus = np.nanmedian(stack, axis=1)
    if np.isfinite(consensus).sum() == 0:
        return None
    return consensus


def _classify_band_cameras(
    band_cam_ids,
    cam_meta,
    anchor_base_rough,
    cam_band_map,
    band_id,
    *,
    late_onset_buffer_days,
    min_quiet_baseline_days,
    min_anchor_overlap_days,
    consensus_start_resid_sigma,
):
    """Bootstrap anchors and identify cameras needing consensus base_rough."""
    if not band_cam_ids:
        return set(), set()

    band_start = min(cam_meta[c]["t_first"] for c in band_cam_ids)

    # Seed anchors from cameras with sufficient quiet baseline on their own masking.
    seed_anchors = {
        cam_id
        for cam_id in band_cam_ids
        if cam_meta[cam_id]["quiet_span"] >= float(min_quiet_baseline_days)
    }

    band_excursion_union: list[tuple[float, float]] = []
    for cam_id in seed_anchors:
        band_excursion_union = _union_intervals(
            band_excursion_union + cam_meta[cam_id]["excursion_intervals"]
        )

    anchors = set(seed_anchors)
    changed = True
    while changed:
        changed = False
        for cam_id in sorted(band_cam_ids, key=lambda c: cam_meta[c]["t_first"]):
            if cam_id in anchors:
                continue
            meta = cam_meta[cam_id]
            quiet_span_band = _quiet_span_before_first_excursion(
                meta["t_first"], meta["t_last"], band_excursion_union
            )
            if quiet_span_band < float(min_quiet_baseline_days):
                continue
            if _time_in_intervals(meta["t_first"], band_excursion_union):
                continue
            has_overlap = any(
                _camera_overlap_days(
                    meta["t_first"],
                    meta["t_last"],
                    cam_meta[other]["t_first"],
                    cam_meta[other]["t_last"],
                )
                >= float(min_anchor_overlap_days)
                for other in band_cam_ids
                if other != cam_id
            )
            if not has_overlap:
                continue
            anchors.add(cam_id)
            changed = True
            band_excursion_union = _union_intervals(
                band_excursion_union + meta["excursion_intervals"]
            )

    # Rebuild union from final anchor set.
    band_excursion_union = []
    for cam_id in anchors:
        band_excursion_union = _union_intervals(
            band_excursion_union + cam_meta[cam_id]["excursion_intervals"]
        )

    # Final anchor qualification: overlap with another camera and not started in union.
    qualified_anchors = set()
    for cam_id in anchors:
        meta = cam_meta[cam_id]
        quiet_span_band = _quiet_span_before_first_excursion(
            meta["t_first"], meta["t_last"], band_excursion_union
        )
        if quiet_span_band < float(min_quiet_baseline_days):
            continue
        if _time_in_intervals(meta["t_first"], band_excursion_union):
            continue
        has_overlap = any(
            _camera_overlap_days(
                meta["t_first"],
                meta["t_last"],
                cam_meta[other]["t_first"],
                cam_meta[other]["t_last"],
            )
            >= float(min_anchor_overlap_days)
            for other in band_cam_ids
            if other != cam_id
        )
        if has_overlap:
            qualified_anchors.add(cam_id)

    needs_consensus: set = set()
    for cam_id in band_cam_ids:
        meta = cam_meta[cam_id]
        late_in_band = (meta["t_first"] - band_start) > float(late_onset_buffer_days)
        if not late_in_band:
            continue

        quiet_span_band = _quiet_span_before_first_excursion(
            meta["t_first"], meta["t_last"], band_excursion_union
        )
        excursion_failure = _time_in_intervals(meta["t_first"], band_excursion_union)
        excursion_failure |= quiet_span_band < float(min_quiet_baseline_days)

        other_anchors = qualified_anchors - {cam_id}
        if other_anchors:
            t_start = np.array([meta["t_first"]], dtype=float)
            consensus_at_start = _build_band_consensus(
                anchor_base_rough,
                band_id,
                t_start,
                other_anchors,
                cam_band_map,
                min_anchor_overlap_days=min_anchor_overlap_days,
            )
            if consensus_at_start is not None and np.isfinite(consensus_at_start[0]):
                mag_start = float(meta["mag_at_first"])
                resid_start = mag_start - float(consensus_at_start[0])
                s0 = meta["mask_s0"]
                if s0 > 0 and (resid_start / s0) > float(consensus_start_resid_sigma):
                    excursion_failure = True

        if not excursion_failure:
            continue

        needs_consensus.add(cam_id)

    return qualified_anchors, needs_consensus


def per_camera_gp_baseline_masked(
    df,
    *,
    dip_sigma_thresh=GP_DIP_SIGMA_THRESH,
    bright_sigma_thresh=GP_BRIGHT_SIGMA_THRESH,
    pad_days=GP_PAD_DAYS,
    S0=BASELINE_S0,
    w0=BASELINE_W0,
    q=BASELINE_Q,
    a1=None,
    rho1=None,
    a2=None,
    rho2=None,
    jitter=BASELINE_JITTER,
    t_col="JD",
    mag_col="mag",
    mag_err_col="error",
    cam_col="camera#",
    band_col="v_g_band",
    min_gp_points=GP_MIN_GP_POINTS,
    add_sigma_eff_col=True,
    sigma_floor=None,
    floor_clip=GP_FLOOR_CLIP,
    floor_iters=GP_FLOOR_ITERS,
    min_floor_points=GP_MIN_FLOOR_POINTS,
    auto_scale_gp=True,
    stiff_scale_fraction=GP_STIFF_SCALE_FRACTION,
    stiff_min_days=GP_STIFF_MIN_DAYS,
    loose_scale_fraction=GP_LOOSE_SCALE_FRACTION,
    loose_min_days=GP_LOOSE_MIN_DAYS,
    late_onset_buffer_days=GP_LATE_ONSET_BUFFER_DAYS,
    min_anchor_overlap_days=GP_MIN_ANCHOR_OVERLAP_DAYS,
    min_quiet_baseline_days=GP_MIN_QUIET_BASELINE_DAYS,
    consensus_start_resid_sigma=GP_CONSENSUS_START_RESID_SIGMA,
    **kwargs,
):
    """Per-camera GP baseline with dip masking (excludes significant dips from fit).

    Cameras that start late within a band and lack sufficient quiet baseline
    borrow anchor consensus base_rough for masking, preventing stiff-GP tracking
    of excursions at camera onset.
    """

    def robust_sigma_floor(resid, mag_err_here, var_here):
        finite0 = np.isfinite(resid) & np.isfinite(mag_err_here) & np.isfinite(var_here)
        if finite0.sum() < max(10, min_floor_points):
            return 0.0

        r = resid[finite0].copy()
        keep = np.ones_like(r, dtype=bool)
        for _ in range(int(max(floor_iters, 1))):
            rr = r[keep]
            if rr.size < max(10, min_floor_points):
                break
            med = float(np.median(rr))
            mad = MAD_SCALE * float(np.median(np.abs(rr - med)))
            mad = max(mad, 1e-12)
            keep = np.abs(r - med) <= float(floor_clip) * mad

        rr = r[keep]
        if rr.size < max(10, min_floor_points):
            rr = r

        s_quiet = MAD_SCALE * float(np.median(np.abs(rr - float(np.median(rr)))))
        s_quiet = max(s_quiet, 1e-12)

        mag_err2_med = float(
            np.median(
                (mag_err_here[finite0][keep] if keep.size == mag_err_here[finite0].size else mag_err_here[finite0]) ** 2
            )
        )
        var_med = float(
            np.median((var_here[finite0][keep] if keep.size == var_here[finite0].size else var_here[finite0]))
        )
        floor2 = max(s_quiet**2 - mag_err2_med - var_med, 0.0)
        return float(np.sqrt(floor2))

    df_out = df.copy()
    out_cols = ("baseline", "base_rough", "resid", "sigma_resid") + (("sigma_eff",) if add_sigma_eff_col else ())
    for col in out_cols:
        if col not in df_out.columns:
            df_out[col] = np.nan

    has_band_col = band_col in df_out.columns
    cam_band_map: dict = {}
    if has_band_col:
        for row_cam, row_band in zip(df_out[cam_col], df_out[band_col]):
            cam_band_map[row_cam] = row_band

    # ---- Pass 1: stiff GP + masking for all cameras ----
    cam_cache: dict = {}
    anchor_base_rough: dict = {}
    cam_meta: dict = {}

    for cam_id, sub in df_out.groupby(cam_col, group_keys=False):
        idx = sub.sort_values(t_col).index
        t = df_out.loc[idx, t_col].to_numpy(float)
        mag = df_out.loc[idx, mag_col].to_numpy(float)
        mag_err = df_out.loc[idx, mag_err_col].to_numpy(float)
        finite = np.isfinite(t) & np.isfinite(mag)
        median_mag = float(np.nanmedian(mag[finite])) if finite.any() else np.nan

        base_rough = _compute_base_rough(
            t,
            mag,
            mag_err,
            finite,
            median_mag,
            auto_scale_gp=auto_scale_gp,
            min_gp_points=min_gp_points,
            stiff_scale_fraction=stiff_scale_fraction,
            stiff_min_days=stiff_min_days,
            S0=S0,
            q=q,
        )
        anchor_base_rough[cam_id] = (t, base_rough)

        flags, keep, excursion_intervals, mask_s0 = _mask_excursions(
            t,
            mag,
            base_rough,
            finite,
            median_mag,
            mag_err=mag_err,
            dip_sigma_thresh=dip_sigma_thresh,
            bright_sigma_thresh=bright_sigma_thresh,
            pad_days=pad_days,
        )

        t_first = float(t[finite].min()) if finite.any() else np.nan
        t_last = float(t[finite].max()) if finite.any() else np.nan
        quiet_span = _quiet_span_before_first_excursion(
            t_first, t_last, excursion_intervals
        )

        cam_cache[cam_id] = {
            "idx": idx,
            "t": t,
            "mag": mag,
            "mag_err": mag_err,
            "finite": finite,
            "median_mag": median_mag,
            "base_rough": base_rough,
            "keep": keep,
        }
        cam_meta[cam_id] = {
            "t_first": t_first,
            "t_last": t_last,
            "quiet_span": quiet_span,
            "excursion_intervals": excursion_intervals,
            "mask_s0": mask_s0,
            "mag_at_first": float(mag[finite][0]) if finite.any() else np.nan,
        }

    # ---- Pass 2: per-band anchor bootstrapping and consensus classification ----
    band_anchors: dict = {}
    needs_consensus: set = set()

    if has_band_col and late_onset_buffer_days > 0:
        for band_id, band_sub in df_out.groupby(band_col, group_keys=False):
            band_cam_ids = list(band_sub[cam_col].unique())
            anchors, consensus_cams = _classify_band_cameras(
                band_cam_ids,
                cam_meta,
                anchor_base_rough,
                cam_band_map,
                band_id,
                late_onset_buffer_days=late_onset_buffer_days,
                min_quiet_baseline_days=min_quiet_baseline_days,
                min_anchor_overlap_days=min_anchor_overlap_days,
                consensus_start_resid_sigma=consensus_start_resid_sigma,
            )
            band_anchors[band_id] = anchors
            needs_consensus.update(consensus_cams)

    def _build_cross_band_consensus(current_band_id, t_target):
        if not has_band_col:
            return None
        best_consensus = None
        best_overlap = 0.0
        for band_id, anchors in band_anchors.items():
            if band_id == current_band_id or not anchors:
                continue
            c = _build_band_consensus(
                anchor_base_rough,
                band_id,
                t_target,
                anchors,
                cam_band_map,
                min_anchor_overlap_days=min_anchor_overlap_days,
            )
            if c is not None:
                overlap = float(np.isfinite(c).sum())
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_consensus = c
        return best_consensus

    # ---- Pass 3: final masking + loose GP per camera ----
    for cam_id, cached in cam_cache.items():
        idx = cached["idx"]
        t = cached["t"]
        mag = cached["mag"]
        mag_err = cached["mag_err"]
        finite = cached["finite"]
        median_mag = cached["median_mag"]
        base_rough = cached["base_rough"].copy()

        if cam_id in needs_consensus:
            band_id = cam_band_map.get(cam_id)
            anchors = band_anchors.get(band_id, set()) if band_id is not None else set()
            consensus = _build_band_consensus(
                anchor_base_rough,
                band_id,
                t,
                anchors,
                cam_band_map,
                min_anchor_overlap_days=min_anchor_overlap_days,
            )
            if consensus is None:
                consensus = _build_cross_band_consensus(band_id, t)
            if consensus is not None:
                base_rough = np.where(np.isfinite(consensus), consensus, base_rough)

        df_out.loc[idx, "base_rough"] = base_rough

        _flags, keep, _intervals, _s0 = _mask_excursions(
            t,
            mag,
            base_rough,
            finite,
            median_mag,
            mag_err=mag_err,
            dip_sigma_thresh=dip_sigma_thresh,
            bright_sigma_thresh=bright_sigma_thresh,
            pad_days=pad_days,
        )

        if keep.sum() < min_gp_points:
            baseline = np.full_like(mag, median_mag, dtype=float)
            resid = mag - baseline

            mag_err_full = mag_err
            floor_here = float(max(sigma_floor, 0.0)) if sigma_floor is not None else float(jitter)
            sigma_eff = np.sqrt(mag_err_full**2 + floor_here**2)
            sigma_resid = resid / sigma_eff

            df_out.loc[idx, "baseline"] = baseline
            df_out.loc[idx, "resid"] = resid
            df_out.loc[idx, "sigma_resid"] = sigma_resid
            if add_sigma_eff_col:
                df_out.loc[idx, "sigma_eff"] = sigma_eff
            continue

        fit_keep = keep & np.isfinite(mag_err) & (mag_err > 0)
        if np.count_nonzero(fit_keep) < min_gp_points:
            baseline = np.full_like(mag, median_mag, dtype=float)
            resid = mag - baseline
            mag_err_full = mag_err
            floor_here = float(max(sigma_floor, 0.0)) if sigma_floor is not None else float(jitter)
            sigma_eff = np.sqrt(mag_err_full**2 + floor_here**2)
            sigma_resid = resid / sigma_eff

            df_out.loc[idx, "baseline"] = baseline
            df_out.loc[idx, "resid"] = resid
            df_out.loc[idx, "sigma_resid"] = sigma_resid
            if add_sigma_eff_col:
                df_out.loc[idx, "sigma_eff"] = sigma_eff
            continue

        t_fit = t[fit_keep]
        mag_fit = mag[fit_keep]
        mag_err_fit = mag_err[fit_keep]

        mean_mag = float(np.mean(mag_fit))
        mag_fit_centered = mag_fit - mean_mag

        if a1 is not None and rho1 is not None and a2 is not None and rho2 is not None:
            k = terms.RealTerm(a=float(a1), c=1.0 / float(rho1)) + terms.RealTerm(a=float(a2), c=1.0 / float(rho2))
        else:
            w0_use = float(w0)
            if auto_scale_gp:
                time_span = float(t_fit.max() - t_fit.min())
                if time_span > 0:
                    target_timescale = max(time_span * float(loose_scale_fraction), float(loose_min_days))
                    w0_scaled = 2.0 * np.pi / target_timescale
                    w0_use = max(w0_scaled, float(w0))
            k = terms.SHOTerm(S0=float(S0), w0=w0_use, Q=float(q))

        try:
            gp = GaussianProcess(k)
            gp.compute(t_fit, diag=mag_err_fit**2)
            mean_prediction, var = gp.predict(mag_fit_centered, t, return_var=True)
        except Exception:
            baseline = np.full_like(mag, median_mag, dtype=float)
            resid = mag - baseline

            mag_err_full = mag_err
            floor_here = float(max(sigma_floor, 0.0)) if sigma_floor is not None else float(jitter)
            sigma_eff = np.sqrt(mag_err_full**2 + floor_here**2)
            sigma_resid = resid / sigma_eff

            df_out.loc[idx, "baseline"] = baseline
            df_out.loc[idx, "resid"] = resid
            df_out.loc[idx, "sigma_resid"] = sigma_resid
            if add_sigma_eff_col:
                df_out.loc[idx, "sigma_eff"] = sigma_eff
            continue

        baseline = np.asarray(mean_prediction, float) + mean_mag
        resid = mag - baseline

        var = np.asarray(var, float)
        var = np.maximum(var, 0.0)

        mag_err_full = mag_err

        if sigma_floor is None:
            floor_here = robust_sigma_floor(resid, mag_err_full, var)
        else:
            floor_here = float(max(sigma_floor, 0.0))

        sigma_eff2 = mag_err_full**2 + floor_here**2 + var
        sigma_eff = np.sqrt(np.maximum(sigma_eff2, 1e-12))
        sigma_resid = resid / sigma_eff

        if add_sigma_eff_col:
            df_out.loc[idx, "sigma_eff"] = sigma_eff

        df_out.loc[idx, "baseline"] = baseline
        df_out.loc[idx, "resid"] = resid
        df_out.loc[idx, "sigma_resid"] = sigma_resid

    return df_out
