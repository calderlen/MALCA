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

from malca.config import DATA_PATH, SKYPATROL_CACHE_DIR
from malca.lightcurve_io import _prepare_lightcurve_df, _resolve_lc_dir
from malca.lightcurve_publication import apply_publication_rcparams, save_publication_figure, FIG_SINGLE_COL_HEATMAP
from scripts.microlensing import fit_candidate_context, _solve_u0_from_A0
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
                files = list(Path(lc_dir).glob(f"*{asas_sn_id}*.dat"))
                if not files:
                    if Path(lc_dir).is_file():
                        lc_path = Path(lc_dir)
                    else:
                        if attempt == max_attempts - 1:
                            return dict(trial_index=trial_index, Amax=Amax, tE=tE, asas_sn_id=asas_sn_id, recovered=False, error="lc_not_found")
                        continue
                else:
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
            error=None
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
    seed: int,
) -> None:
    _GLOBAL["control_ids"] = control_ids
    _GLOBAL["control_dirs"] = control_dirs
    _GLOBAL["Amax_range"] = Amax_range
    _GLOBAL["tE_range"] = tE_range
    _GLOBAL["mag_err_poly"] = mag_err_poly
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
        lc_dir = _resolve_lc_dir(row)
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
        lc_dir = _resolve_lc_dir(row)
        try:
            from malca.fetch import download_lightcurve_by_id
            from malca.config import SKYPATROL_CACHE_DIR
            
            if not lc_dir:
                lc_path, _ = download_lightcurve_by_id(asas_sn_id, cache_dir=SKYPATROL_CACHE_DIR)
                if lc_path is None:
                    continue
            else:
                files = list(Path(lc_dir).glob(f"*{asas_sn_id}*.dat"))
                if files:
                    lc_path = files[0]
                elif Path(lc_dir).is_file():
                    lc_path = Path(lc_dir)
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

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_HEATMAP)
    
    # Plot heatmap
    X, Y = np.meshgrid(x_edge, y_edge)
    cmap = plt.cm.viridis
    cmap.set_bad(color='0.9')
    mesh = ax.pcolormesh(X, Y, stat.T, cmap=cmap, vmin=0, vmax=1)
    
    # Add contours
    mask = ~np.isnan(stat.T)
    if np.any(mask):
        xc = (x_edge[:-1] + x_edge[1:]) / 2
        yc = (y_edge[:-1] + y_edge[1:]) / 2
        XC, YC = np.meshgrid(xc, yc)
        ax.contour(
            XC,
            YC,
            stat.T,
            levels=[0.1, 0.5, 0.9],
            colors='white',
            alpha=0.8,
            linewidths=1.0,
            linestyles=[':', '--', '-']
        )
    
    cbar = fig.colorbar(mesh, ax=ax, label='Recovery Efficiency')
    
    # Set tick labels to original scale
    ax.set_xlabel(r'$t_E$ (days)')
    ax.set_ylabel(r'$A_{max}$')

    x_ticks = np.linspace(log_tE.min(), log_tE.max(), 5)
    y_ticks = np.linspace(log_Amax.min(), log_Amax.max(), 5)
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{10**x:.1f}" for x in x_ticks])
    
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{10**y:.1f}" for y in y_ticks])

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

def main():
    parser = argparse.ArgumentParser(description='Run microlensing injection-recovery and generate efficiency map')
    parser.add_argument('--manifest', type=str, required=True,
                        help='Path to parquet or csv manifest of clean lightcurves')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output parquet file')
    parser.add_argument('--trials', type=int, default=1000,
                        help='Number of injection trials to run')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of workers for multiprocessing')
    parser.add_argument('--n-stars', type=float, default=1e7,
                        help='Number of monitored stars (for Gamma calculations)')
    parser.add_argument('--duration', type=float, default=5.0,
                        help='Survey duration in years (for Gamma calculations)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output')
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if manifest_path.suffix == '.parquet':
        df_manifest = pd.read_parquet(manifest_path)
    else:
        df_manifest = pd.read_csv(manifest_path)

    output_parquet_path = Path(args.output)
    output_plot_path = output_parquet_path.with_suffix('.pdf')

    run_microlensing_injection_recovery(
        control_sample=df_manifest,
        total_trials=args.trials,
        workers=args.workers,
        output_path=output_parquet_path,
        overwrite=args.overwrite,
    )

    print("Generating efficiency map...")
    df_results = pd.read_parquet(output_parquet_path)
    plot_efficiency_map(df_results, output_plot_path)
    calculate_event_rate(df_results, args.n_stars, args.duration)

if __name__ == '__main__':
    main()
