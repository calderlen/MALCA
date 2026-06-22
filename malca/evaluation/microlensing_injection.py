#!/usr/bin/env python3
"""
Microlensing Injection-Recovery Pipeline

Injects synthetic Paczynski microlensing events into clean light curves and measures
the pipeline's detection efficiency as a function of tE and Amax.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

from malca.config import SKYPATROL_CACHE_DIR
from malca.lightcurve_publication import apply_publication_rcparams, save_publication_figure, FIG_SINGLE_COL_HEATMAP, scaled_publication_text_sizes

import sys
sys.path.append(str(Path.cwd()))
from scripts.microlensing import fit_candidate_context, _solve_u0_from_A0, _prepare_lightcurve_df
from malca.evaluation.injection import _resolve_lc_path
from malca.config import (
    INJECTION_MAX_ATTEMPTS,
    INJECTION_MAG_LO,
    INJECTION_MAG_HI,
    DEFAULT_OUTPUT_DIR,
)
from malca.evaluation.injection import (
    ParquetAppendWriter,
    _write_checkpoint,
    _read_checkpoint,
    estimate_magnitude_error_polynomial,
    _get_id_col,
)

from scripts.microlensing import (
    fit_candidate_context,
    _prepare_lightcurve_df,
)

_GLOBAL: dict[str, object] = {}

def _solve_u0_from_A0(A0: float) -> float:
    """Find u0 corresponding to maximum magnification A0."""
    A_curr = float(A0)
    if A_curr <= 1.0:
        return float('inf')
    # Use exact formula if possible: A = (u^2 + 2)/(u * sqrt(u^2+4))
    # -> u^4 + 4u^2 - 4 / (A^2 - 1) = 0
    # -> u^2 = -2 + sqrt(4 + 4/(A^2-1))
    val = -2.0 + np.sqrt(4.0 + 4.0 / (A_curr**2 - 1.0))
    if val > 0:
        return float(np.sqrt(val))
    return 1e-8


def inject_paczynski(
    df_lc: pd.DataFrame,
    t_center: float,
    tE: float,
    Amax: float,
    mag_err_poly: np.poly1d | None = None,
    rng: np.random.Generator | None = None,
    mag_col: str = "mag",
    time_col: str = "JD",
    err_col: str = "error",
) -> pd.DataFrame:
    """Inject a Paczynski microlensing profile into a light curve."""
    df_out = df_lc.copy()
    if df_out.empty:
        return df_out
        
    t = df_out[time_col].values
    mag_old = df_out[mag_col].values
    
    u0 = _solve_u0_from_A0(float(Amax))
    # Calculate magnification A(t)
    u_sq = u0**2 + ((t - t_center) / tE)**2
    u = np.sqrt(u_sq)
    A = (u_sq + 2.0) / (u * np.sqrt(u_sq + 4.0))
    
    # Brightening -> negative magnitude offset
    dip_profile = -2.5 * np.log10(A)
    
    # Calculate noise to add
    if mag_err_poly is not None:
        sigma_i = np.asarray(mag_err_poly(mag_old), dtype=float)
    else:
        sigma_i = df_out[err_col].values.astype(float)
        
    valid_mask = np.isfinite(sigma_i) & (sigma_i > 0)
    if valid_mask.any():
        fallback = np.nanmedian(sigma_i[valid_mask])
    else:
        fallback = 0.01
    sigma_i = np.where(valid_mask, sigma_i, fallback)

    rng = np.random.default_rng() if rng is None else np.random.default_rng()
    noise = rng.normal(0.0, sigma_i, size=len(t))
    df_out[mag_col] = mag_old + dip_profile + noise
    
    return df_out


def _simulate_microlensing_trial(
    trial_index: int,
    *,
    control_ids: np.ndarray,
    control_dirs: np.ndarray,
    Amax_range: tuple[float, float],
    tE_range: tuple[float, float],
    mag_err_poly: np.poly1d | None,
    measure_pre_injection: bool,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed + int(trial_index))
    
    # Amax: Log-Uniform
    log_Amax_min = np.log10(Amax_range[0])
    log_Amax_max = np.log10(Amax_range[1])
    Amax = 10 ** rng.uniform(log_Amax_min, log_Amax_max)
        
    # tE: Log-Uniform
    log_tE_min = np.log10(tE_range[0])
    log_tE_max = np.log10(tE_range[1])
    tE = 10 ** rng.uniform(log_tE_min, log_tE_max)

    max_attempts = INJECTION_MAX_ATTEMPTS
    for attempt in range(max_attempts):
        control_idx = int(rng.integers(0, len(control_ids)))
        asas_sn_id = str(control_ids[control_idx])
        lc_dir = Path(str(control_dirs[control_idx]))

        try:
            # We use _prepare_lightcurve_df to get the same clean single-band dataframe
            # that the pipeline expects. We pass the directory or file path.
            # Assuming lc_dir has the .dat files inside.
            from malca.fetch import download_lightcurve_by_id
            from malca.config import SKYPATROL_CACHE_DIR
            
            if not lc_dir or str(lc_dir) == '.':
                lc_path, _ = download_lightcurve_by_id(asas_sn_id, cache_dir=SKYPATROL_CACHE_DIR)
                if lc_path is None:
                    if attempt == max_attempts - 1:
                        return dict(trial_index=trial_index, Amax=Amax, tE=tE, asas_sn_id=asas_sn_id, recovered=False, error="lc_not_found")
                    continue
            else:
                lc_path = Path(lc_dir)
                if not lc_path.is_file():
                    # Fallback to glob only if it's a directory
                    files = list(lc_path.glob(f"*{asas_sn_id}*.dat"))
                    if not files:
                        if attempt == max_attempts - 1:
                            return dict(trial_index=trial_index, Amax=Amax, tE=tE, asas_sn_id=asas_sn_id, recovered=False, error="lc_not_found")
                        continue
                    lc_path = files[0]

            df_lc, band_label = _prepare_lightcurve_df(lc_path, prefer_g_band=True)
            if df_lc.empty or len(df_lc) < 20:
                if attempt == max_attempts - 1:
                    return dict(trial_index=trial_index, Amax=Amax, tE=tE, asas_sn_id=asas_sn_id, recovered=False, error="empty_or_short_lc_max_retries")
                continue

            median_mag = float(np.nanmedian(df_lc["mag"].values))
            if median_mag < INJECTION_MAG_LO or median_mag > INJECTION_MAG_HI:
                if attempt == max_attempts - 1:
                     return dict(trial_index=trial_index, Amax=Amax, tE=tE, asas_sn_id=asas_sn_id, recovered=False, error="magnitude_out_of_range")
                continue
            
            break
            
        except Exception as exc:
            if attempt == max_attempts - 1:
                return dict(trial_index=trial_index, Amax=Amax, tE=tE, asas_sn_id=asas_sn_id, recovered=False, error=str(exc))
            continue

    try:
        t_min = float(df_lc["JD"].min())
        t_max = float(df_lc["JD"].max())
        if not np.isfinite(t_min) or not np.isfinite(t_max) or (t_max - t_min <= 4 * tE):
            return dict(trial_index=trial_index, Amax=Amax, tE=tE, asas_sn_id=asas_sn_id, recovered=False, error="invalid_time_range")

        # Measure pre-injection detection rate if requested
        pre_injection_result = {}
        if measure_pre_injection:
            pre_context = {
                'candidate_id': asas_sn_id,
                'asas_sn_id': asas_sn_id,
                'row': {},
                'payload': {
                    'candidate_id': asas_sn_id,
                    'ra_deg': 0.0,
                    'dec_deg': 0.0,
                },
                'lc_path': Path("pre_injected.dat"),
                'df': df_lc,
                'band_label': band_label,
            }
            pre_fit = fit_candidate_context(pre_context)
            if pre_fit is not None:
                p_summary = pre_fit.get('summary', {})
                p_fit_ok = bool(p_summary.get('fit_ok', False))
                p_best_model = p_summary.get('best_model')
                p_paczynski_summary = pre_fit.get('models', {}).get('paczynski', {})
                p_reduced_chi2 = p_paczynski_summary.get('reduced_chi2', float('inf'))
                p_is_paczynski = (p_best_model == 'paczynski')
                p_good_fit = p_fit_ok and p_reduced_chi2 < 10.0
                pre_injection_result = {
                    "pre_injection_recovered": p_is_paczynski and p_good_fit,
                    "pre_injection_fit_ok": p_fit_ok,
                    "pre_injection_best_model": p_best_model
                }
            else:
                pre_injection_result = {
                    "pre_injection_recovered": False,
                    "pre_injection_fit_ok": False,
                    "pre_injection_error": "fit_returned_none"
                }

        # Inject curve inside the observing window, leaving 2 tE buffer at edges
        t_center = rng.uniform(t_min + 2 * tE, t_max - 2 * tE)

        df_injected = inject_paczynski(
            df_lc,
            t_center,
            tE,
            Amax,
            mag_err_poly,
            rng=rng,
        )
        
        # Build context for fit_candidate_context
        context = {
            'candidate_id': asas_sn_id,
            'asas_sn_id': asas_sn_id,
            'row': {},
            'payload': {
                'candidate_id': asas_sn_id,
                'ra_deg': 0.0,
                'dec_deg': 0.0,
            },
            'lc_path': Path("injected.dat"),
            'df': df_injected,
            'band_label': band_label,
        }

        fit_result = fit_candidate_context(context)
        
        if fit_result is None:
             return dict(trial_index=trial_index, Amax=Amax, tE=tE, t_center=t_center, asas_sn_id=asas_sn_id, recovered=False, error="fit_returned_none")
             
        summary = fit_result.get('summary', {})
        
        fit_ok = bool(summary.get('fit_ok', False))
        best_model = summary.get('best_model')
        reduced_chi2 = float(summary.get('paczynski_reduced_chi2', np.nan))
        recovered_tE = float(summary.get('raw_paczynski_tE_days', np.nan))
        
        is_paczynski = best_model == 'paczynski'
        good_fit = fit_ok and reduced_chi2 < 10.0
        recovered = is_paczynski and good_fit
        
        return dict(
            trial_index=trial_index,
            Amax=Amax,
            tE=tE,
            t_center=float(t_center),
            median_mag=median_mag,
            asas_sn_id=asas_sn_id,
            recovered=recovered,
            fit_ok=fit_ok,
            best_model=str(best_model),
            reduced_chi2=reduced_chi2,
            recovered_tE=recovered_tE,
            n_points=len(df_injected),
            error=None,
            **pre_injection_result
        )
    except Exception as exc:
        return dict(
            trial_index=trial_index,
            Amax=Amax,
            tE=tE,
            asas_sn_id=asas_sn_id,
            recovered=False,
            error=str(exc),
        )


def _init_worker(
    control_ids: np.ndarray,
    control_dirs: np.ndarray,
    Amax_range: tuple[float, float],
    tE_range: tuple[float, float],
    mag_err_poly: np.poly1d | None,
    measure_pre_injection: bool,
    seed: int,
) -> None:
    _GLOBAL["control_ids"] = control_ids
    _GLOBAL["control_dirs"] = control_dirs
    _GLOBAL["Amax_range"] = Amax_range
    _GLOBAL["tE_range"] = tE_range
    _GLOBAL["mag_err_poly"] = mag_err_poly
    _GLOBAL["measure_pre_injection"] = measure_pre_injection
    _GLOBAL["seed"] = seed


def _process_trial_batch(trial_indices: list[int]) -> list[dict]:
    results = []
    for trial_index in trial_indices:
        results.append(
            _simulate_microlensing_trial(
                trial_index,
                control_ids=_GLOBAL["control_ids"],
                control_dirs=_GLOBAL["control_dirs"],
                Amax_range=_GLOBAL["Amax_range"],
                tE_range=_GLOBAL["tE_range"],
                mag_err_poly=_GLOBAL["mag_err_poly"],
                measure_pre_injection=bool(_GLOBAL["measure_pre_injection"]),
                seed=int(_GLOBAL["seed"]),
            )
        )
    return results


def run_microlensing_injection_recovery(
    control_sample: pd.DataFrame,
    *,
    total_trials: int = 1000,
    Amax_range: tuple[float, float] = (1.05, 100.0),
    tE_range: tuple[float, float] = (1.0, 300.0),
    measure_pre_injection: bool = False,
    mag_err_order: int = 5,
    mag_err_sample: int = 100,
    seed: int = 42,
    workers: int = 1,
    task_size: int = 10,
    checkpoint_interval: int = 1000,
    chunk_size: int = 100,
    output_path: Path | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = True,
    overwrite: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame | None:
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists() and overwrite and not resume:
            output_path.unlink()
        if output_path.exists() and not resume and not overwrite:
            raise SystemExit(f"Output exists: {output_path} (use --overwrite or --no-resume)")

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
    elif output_path is not None:
        checkpoint_path = output_path.with_name(f"{output_path.stem}_PROCESSED.txt")

    if checkpoint_path and checkpoint_path.exists() and overwrite and not resume:
        checkpoint_path.unlink()

    start_index = 0
    if resume and checkpoint_path and checkpoint_path.exists():
        last = _read_checkpoint(checkpoint_path)
        if last is not None:
            start_index = int(last) + 1

    if start_index >= total_trials:
        print("All trials already completed per checkpoint.")
        return None

    id_col = _get_id_col(control_sample)
    control_ids = control_sample[id_col].astype(str).to_numpy()
    control_dirs = []
    for _, row in control_sample.iterrows():
        lc_dir = _resolve_lc_path(row)
        if lc_dir is None:
            control_dirs.append("")
        else:
            control_dirs.append(str(lc_dir))
    control_dirs = np.asarray(control_dirs, dtype=object)

    if len(control_ids) == 0:
        raise SystemExit("Control sample is empty.")

    print("Loading control sample light curves for error polynomial...")
    lc_sample = []
    for idx, row in control_sample.iterrows():
        if idx >= mag_err_sample:
            break
        asas_sn_id = str(row[id_col])
        lc_dir = _resolve_lc_path(row)
        try:
            from malca.fetch import download_lightcurve_by_id
            from malca.config import SKYPATROL_CACHE_DIR
            
            if not lc_dir:
                lc_path, _ = download_lightcurve_by_id(asas_sn_id, cache_dir=SKYPATROL_CACHE_DIR)
                if lc_path is None:
                    continue
            else:
                lc_path = Path(lc_dir)
                if not lc_path.is_file():
                    files = list(lc_path.glob(f"*{asas_sn_id}*.dat"))
                    if files:
                        lc_path = files[0]
                    else:
                        continue
                    
            df_lc, _ = _prepare_lightcurve_df(lc_path)
            if not df_lc.empty:
                lc_sample.append(df_lc)
        except Exception:
            continue

    print(f"Fitting {mag_err_order}th-order polynomial to magnitude errors...")
    mag_err_poly = estimate_magnitude_error_polynomial(lc_sample, order=mag_err_order)

    writer = ParquetAppendWriter(output_path) if output_path else None
    results: list[dict] = []

    pbar = tqdm(total=total_trials, initial=start_index, disable=not show_progress)

    def flush_results(is_final: bool = False) -> None:
        nonlocal results
        if not results:
            return
        if writer is None:
            return
        writer.write_chunk(results)
        if is_final:
            writer.close()
        results = []

    if workers <= 1:
        for trial_index in range(start_index, total_trials):
            res = _simulate_microlensing_trial(
                trial_index,
                control_ids=control_ids,
                control_dirs=control_dirs,
                Amax_range=Amax_range,
                tE_range=tE_range,
                mag_err_poly=mag_err_poly,
                measure_pre_injection=measure_pre_injection,
                seed=seed,
            )
            results.append(res)
            pbar.update(1)
            if chunk_size and len(results) >= chunk_size:
                flush_results()
            if checkpoint_path and (trial_index + 1) % checkpoint_interval == 0:
                flush_results()
                _write_checkpoint(checkpoint_path, trial_index)
        flush_results(is_final=True)
        if checkpoint_path:
            _write_checkpoint(checkpoint_path, total_trials - 1)
        pbar.close()
        return None if output_path else pd.DataFrame(results)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(
            control_ids,
            control_dirs,
            Amax_range,
            tE_range,
            mag_err_poly,
            measure_pre_injection,
            seed,
        ),
    ) as ex:
        for batch_start in range(start_index, total_trials, checkpoint_interval):
            batch_end = min(batch_start + checkpoint_interval, total_trials)
            batch_indices = list(range(batch_start, batch_end))
            tasks = [batch_indices[i:i + task_size] for i in range(0, len(batch_indices), task_size)]

            futures = {ex.submit(_process_trial_batch, task): task for task in tasks}
            for fut in as_completed(futures):
                batch_results = fut.result()
                results.extend(batch_results)
                pbar.update(len(batch_results))
                if chunk_size and len(results) >= chunk_size:
                    flush_results()

            flush_results()
            if checkpoint_path:
                _write_checkpoint(checkpoint_path, batch_end - 1)

    flush_results(is_final=True)
    if checkpoint_path:
        _write_checkpoint(checkpoint_path, total_trials - 1)
    pbar.close()
    return None if output_path else pd.DataFrame(results)


def plot_efficiency_map(
    df: pd.DataFrame,
    out_path: Path,
    bins_tE: int = 15,
    bins_Amax: int = 15
):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from astropy.convolution import convolve, Gaussian2DKernel

    apply_publication_rcparams(plt)
    df_clean = df.dropna(subset=['tE', 'Amax', 'recovered'])

    log_tE = np.log10(df_clean['tE'].values)
    log_Amax = np.log10(df_clean['Amax'].values)
    recovered = df_clean['recovered'].values.astype(float)

    # Bin the data
    tE_bins = np.linspace(log_tE.min(), log_tE.max(), bins_tE + 1)
    Amax_bins = np.linspace(log_Amax.min(), log_Amax.max(), bins_Amax + 1)

    stat, x_edge, y_edge, _ = binned_statistic_2d(
        log_tE, log_Amax, recovered,
        statistic='mean', bins=[tE_bins, Amax_bins]
    )
    
    # Smooth with NaN handling
    kernel = Gaussian2DKernel(x_stddev=1.0)
    smoothed_eff = convolve(stat.T, kernel, boundary='extend', preserve_nan=False)

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_HEATMAP)
    text = scaled_publication_text_sizes(FIG_SINGLE_COL_HEATMAP)
    
    xc = (x_edge[:-1] + x_edge[1:]) / 2
    yc = (y_edge[:-1] + y_edge[1:]) / 2
    
    cmap = plt.cm.cividis
    cmap.set_bad(color='0.9')
    
    levels_contourf = np.linspace(0.0, 1.0, 100)
    im = ax.contourf(
        xc,
        yc,
        smoothed_eff,
        levels=levels_contourf,
        cmap=cmap,
        extend='both'
    )
    
    # Rasterize
    try:
        for c in im.collections:
            c.set_edgecolor("face")
            c.set_rasterized(True)
    except AttributeError:
        im.set_edgecolor("face")
        im.set_rasterized(True)
    
    # Add contours
    mask = ~np.isnan(stat.T)
    if np.any(mask):
        try:
            cs = ax.contour(
                xc,
                yc,
                smoothed_eff,
                levels=[0.5, 0.9, 0.99],
                colors='black',
                alpha=0.9,
                linewidths=0.6
            )
            
            # Find the longest continuous segment for each contour level to place exactly one label
            label_locations = []
            for p in cs.get_paths():
                polys = p.to_polygons()
                if not polys:
                    continue
                longest_poly = max(polys, key=lambda poly: len(poly))
                mid_idx = len(longest_poly) // 2
                midpoint = longest_poly[mid_idx]
                label_locations.append((midpoint[0], midpoint[1]))
            
            if label_locations:
                texts = ax.clabel(cs, inline=True, inline_spacing=4, fontsize=8, fmt='%g', manual=label_locations)
                for t in texts:
                    t.set_rotation(0)
        except Exception:
            pass
        
    # Marginal axes
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", size="20%", pad=0.15, sharex=ax)
    ax_histy = divider.append_axes("left", size="20%", pad=0.15, sharey=ax)
    
    with np.errstate(invalid='ignore'):
        eff_x = np.nanmean(smoothed_eff, axis=0) # avg over Amax
        eff_y = np.nanmean(smoothed_eff, axis=1) # avg over tE
        
    left_lim = min(0.0, log_tE.min())
    bottom_lim = min(0.0, log_Amax.min())
    
    ax.set_xlim(left=left_lim, right=log_tE.max())
    ax.set_ylim(bottom=bottom_lim, top=log_Amax.max())
    
    # ax_histx (top)
    ax_histx.plot(xc, eff_x, color="black", lw=0.6)
    ax_histx.set_ylim(0, 1)
    ax_histx.set_yticks([0, 1])
    ax_histx.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=False, which="both")
    ax_histx.yaxis.tick_right()
    ax_histx.yaxis.set_label_position("right")
    ax_histx.set_ylabel("Efficiency", fontsize=text["label"]*0.85)
    ax_histx.tick_params(axis="y", labelsize=text["label"]*0.75)
    
    # ax_histy (left)
    ax_histy.plot(eff_y, yc, color="black", lw=0.6)
    ax_histy.set_xlim(0, 1)
    ax_histy.set_xticks([0, 1])
    ax_histy.invert_xaxis()
    ax_histy.tick_params(axis="y", labelleft=True, labelright=False)
    ax_histy.xaxis.tick_top()
    ax_histy.xaxis.set_label_position("top")
    ax_histy.set_xlabel("Efficiency", fontsize=text["label"]*0.85)
    ax_histy.tick_params(axis="x", labelsize=text["label"]*0.75)
    ax_histy.set_ylabel(r'$A_{max}$', fontsize=text["label"])
    
    ax.set_xlabel(r'$t_E$ [days]', fontsize=text["label"])
    ax.tick_params(axis="y", labelleft=False)
    
    # Manually configure logarithmic ticks on the linear axes
    import matplotlib.ticker as ticker
    def set_log_ticks_on_linear_axis(axis_obj, vmin, vmax):
        # Only create major ticks strictly within the plot limits (or slightly outside to allow margin)
        major_ticks = np.arange(np.floor(vmin), np.ceil(vmax) + 1)
        # Filter out major ticks that are way outside the limits to avoid extending the axis
        major_ticks = [x for x in major_ticks if x <= vmax + 0.1 and x >= vmin - 0.1]
        
        axis_obj.set_ticks(major_ticks)
        axis_obj.set_ticklabels([rf"$10^{{{int(x)}}}$" for x in major_ticks])
        
        minor_ticks = []
        for power in np.arange(np.floor(vmin)-1, np.ceil(vmax)+1):
            for mult in range(2, 10):
                val = np.log10(mult * 10**power)
                if vmin <= val <= vmax:
                    minor_ticks.append(val)
        axis_obj.set_ticks(minor_ticks, minor=True)

    set_log_ticks_on_linear_axis(ax.xaxis, left_lim, log_tE.max())
    set_log_ticks_on_linear_axis(ax_histy.yaxis, bottom_lim, log_Amax.max())
    
    # Add a secondary y-axis for magnitude drop (Delta m)
    def logA_to_dm(logA):
        return 2.5 * logA

    def dm_to_logA(dm):
        return dm / 2.5

    secax = ax.secondary_yaxis('right', functions=(logA_to_dm, dm_to_logA))
    dm_ticks = np.array([0.1, 0.5, 1.0, 2.0, 3.0, 5.0])
    # Only keep dm_ticks that fall within the Y axis range
    dm_ticks = [dm for dm in dm_ticks if dm_to_logA(dm) <= log_Amax.max()]
    secax.set_yticks(dm_ticks)
    secax.set_yticklabels([f"{dm:.1f}" for dm in dm_ticks])
    secax.set_ylabel(r"$\Delta m$ [mag]", fontsize=text["label"])
    secax.tick_params(axis="y", labelsize=text["label"]*0.75)

    # STRICTLY set the axis limits at the very end to prevent matplotlib from autoscaling to the ticks!
    ax.set_xlim(left=log_tE.min(), right=log_tE.max())
    ax.set_ylim(bottom=log_Amax.min(), top=log_Amax.max())
    ax_histx.set_xlim(left=log_tE.min(), right=log_tE.max())
    ax_histy.set_ylim(bottom=log_Amax.min(), top=log_Amax.max())

    cax = divider.append_axes("right", size="7%", pad=0.4)
    cbar = plt.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    import matplotlib.ticker as ticker
    cax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')
    cbar.set_label("Efficiency", fontsize=text["label"], labelpad=8)
    cbar.ax.tick_params(labelsize=text["label"]*0.6)
    
    fig.tight_layout()
    save_publication_figure(fig, out_path, dpi=300)
    print(f"Efficiency map saved to {out_path}")

def calculate_event_rate(df: pd.DataFrame, n_stars_monitored: float, duration_years: float):
    if df.empty:
        return
        
    overall_efficiency = df['recovered'].mean()
    n_observed_events = 30 # User specified ~30 observed events
    
    if overall_efficiency > 0:
        true_events = n_observed_events / overall_efficiency
        event_rate = true_events / (n_stars_monitored * duration_years)
        print(f"\n--- Microlensing Event Rate Estimate ---")
        print(f"Overall Recovery Efficiency: {overall_efficiency:.2%}")
        print(f"Observed Events: {n_observed_events}")
        print(f"Estimated True Events: {true_events:.1f}")
        print(f"Stars Monitored: {n_stars_monitored:.1e}")
        print(f"Duration (years): {duration_years:.1f}")
        print(f"Estimated Event Rate (Gamma): {event_rate:.2e} events/star/yr")
        print(f"----------------------------------------\n")
    else:
        print("Overall efficiency is 0, cannot calculate event rate.")

from datetime import datetime
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run microlensing injection-recovery and generate efficiency map')
    parser.add_argument('--manifest', type=str, default=str(DEFAULT_OUTPUT_DIR / "lc_manifest_all.parquet"),
                        help='Path to parquet or csv manifest of clean lightcurves (defaults to standard cluster manifest)')
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "microlensing_injection",
                        help=f"Base output directory (default: {DEFAULT_OUTPUT_DIR / 'microlensing_injection'})")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Optional tag to append to run directory name")
    parser.add_argument('--output', type=Path, default=None,
                        help='Override Parquet output path (default: <out-dir>/<timestamp>/microlensing_results.parquet)')
    parser.add_argument('--trials', type=int, default=1000,
                        help='Number of injection trials to run')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of workers for multiprocessing')
    parser.add_argument('--n-stars', type=float, default=17e6,
                        help='Number of monitored stars (for Gamma calculations)')
    parser.add_argument('--duration', type=float, default=12.0,
                        help='Survey duration in years (for Gamma calculations)')
    parser.add_argument('--amp-min', type=float, default=1.05, help='Minimum injected magnification (Amax)')
    parser.add_argument('--amp-max', type=float, default=100.0, help='Maximum injected magnification (Amax)')
    parser.add_argument('--dur-min', type=float, default=1.0, help='Minimum injected Einstein time (tE in days)')
    parser.add_argument('--dur-max', type=float, default=500.0, help='Maximum injected Einstein time (tE in days)')
    parser.add_argument('--measure-pre-injection', action='store_true', help='Measure fit before injecting to establish clean baseline')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output')
    parser.add_argument('--plot-only', action='store_true',
                        help='Only generate the plot from an existing parquet file')
    parser.add_argument('--bins-te', type=int, default=15, help='Number of bins for tE in plot')
    parser.add_argument('--bins-amax', type=int, default=15, help='Number of bins for Amax in plot')
    args = parser.parse_args()

    # Set up output paths with timestamped run directory
    base_out_dir = Path(args.out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.run_tag}" if args.run_tag else timestamp
    
    if args.output:
        output_parquet_path = Path(args.output)
        run_dir = output_parquet_path.parent
    else:
        run_dir = base_out_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        output_parquet_path = run_dir / "microlensing_results.parquet"

    output_plot_path = output_parquet_path.with_suffix('.pdf')

    # Save run parameters to JSON
    run_params_file = run_dir / "run_params.json"
    run_params = vars(args).copy()
    for key, value in run_params.items():
        if isinstance(value, Path):
            run_params[key] = str(value)
    with open(run_params_file, "w") as f:
        json.dump(run_params, f, indent=2, default=str)

    # Create/update 'latest' symlink only if we created a new run_dir
    if not args.output:
        latest_link = base_out_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        try:
            latest_link.symlink_to(run_name)
        except Exception as e:
            pass # Symlinks might fail on some filesystems

    if not args.plot_only:
        manifest_path = Path(args.manifest)
        if manifest_path.suffix == '.parquet':
            df_manifest = pd.read_parquet(manifest_path)
        else:
            df_manifest = pd.read_csv(manifest_path)

        run_microlensing_injection_recovery(
            control_sample=df_manifest,
            total_trials=args.trials,
            workers=args.workers,
            Amax_range=(args.amp_min, args.amp_max),
            tE_range=(args.dur_min, args.dur_max),
            measure_pre_injection=args.measure_pre_injection,
            output_path=output_parquet_path,
            overwrite=args.overwrite,
        )

    print(f"Generating efficiency map in {output_plot_path}...")
    df_results = pd.read_parquet(output_parquet_path)
    plot_efficiency_map(df_results, output_plot_path, bins_tE=args.bins_te, bins_Amax=args.bins_amax)
    calculate_event_rate(df_results, args.n_stars, args.duration)

if __name__ == '__main__':
    main()
