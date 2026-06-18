from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/malca-matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from malca.lightcurve_publication import (
    apply_publication_rcparams,
    figsize_two_col_grid,
)

apply_publication_rcparams(plt)

from malca.baseline import per_camera_gp_baseline_masked, phase_template_baseline
from malca.config import (
    BASELINE_JITTER,
    BASELINE_Q,
    BASELINE_S0,
    BASELINE_W0,
    DEFAULT_OUTPUT_DIR,
    LOGBF_THRESHOLD_DIP,
    MAG_POINTS,
    PARQUET_OUTPUT_COMPRESSION,
    P_POINTS,
    RUN_MAX_GAP_POINTS,
    RUN_MIN_POINTS,
    SIGNIFICANCE_THRESHOLD,
)
from malca.stv.events import build_runs, filter_runs, score_events_bayesian, summarize_kept_runs
from malca.lightcurve_publication import (
    plot_lightcurve_panel,
    plot_phase_panel,
    plot_residual_panel,
    style_publication_axis,
)
from malca.stv.triggering import posterior_probability_threshold, resolve_trigger_indices
from malca.utils import clean_lc

from malca.evaluation.periodic_branch_simulation_benchmark import (
    DEFAULT_MODE_NAMES,
    PERIODIC_BRANCH_MODE_SPECS,
    add_metric_bins,
    generate_trial_design,
    simulate_periodic_lightcurve,
    _mode_selected_period,
)


DEFAULT_POSTERIOR_PROBABILITY_THRESHOLDS: tuple[float, ...] = (
    0.90,
    0.95,
    0.99,
    0.999,
    0.9999,
    0.99999,
    posterior_probability_threshold(SIGNIFICANCE_THRESHOLD),
)

DEFAULT_LOGBF_THRESHOLDS: tuple[float, ...] = (
    0.0,
    1.0,
    2.0,
    3.0,
    LOGBF_THRESHOLD_DIP,
    7.0,
    10.0,
    15.0,
    20.0,
)


@dataclass
class TriggerModeBenchmarkConfig:
    output_base_dir: Path = DEFAULT_OUTPUT_DIR / "diagnostics" / "trigger_mode_simulation_benchmark"
    run_tag: str | None = None
    n_trials: int = 12000
    seed: int = 20260514
    workers: int = 8
    show_progress: bool = True
    force: bool = False
    mode_names: tuple[str, ...] = DEFAULT_MODE_NAMES

    # Same synthetic population mix as the periodic branch benchmark.
    control_fraction: float = 0.10
    small_dip_fraction: float = 0.50
    medium_dip_fraction: float = 0.32
    broad_dip_fraction: float = 0.08

    # Event scoring. Scores are computed once; trigger thresholds are swept
    # from the shared LOO posterior probability and local log-BF arrays.
    logbf_threshold_dip: float = LOGBF_THRESHOLD_DIP
    significance_threshold: float = SIGNIFICANCE_THRESHOLD
    posterior_probability_thresholds: tuple[float, ...] = DEFAULT_POSTERIOR_PROBABILITY_THRESHOLDS
    logbf_thresholds: tuple[float, ...] = DEFAULT_LOGBF_THRESHOLDS
    p_points: int = P_POINTS
    mag_points: int = MAG_POINTS
    run_min_points: int = RUN_MIN_POINTS
    run_max_gap_points: int = RUN_MAX_GAP_POINTS
    run_max_gap_days: float | None = None
    run_min_duration_days: float = 0.0

    # Baseline kwargs for parity with events.py defaults.
    baseline_s0: float = BASELINE_S0
    baseline_w0: float = BASELINE_W0
    baseline_q: float = BASELINE_Q
    baseline_jitter: float = BASELINE_JITTER
    baseline_sigma_floor: float | None = None


@dataclass
class TriggerModeBenchmarkRun:
    config: TriggerModeBenchmarkConfig
    run_dir: Path
    trial_design: pd.DataFrame
    score_results: pd.DataFrame
    trigger_results: pd.DataFrame
    summary_slices: dict[str, pd.DataFrame]
    pairwise_production: pd.DataFrame


def make_run_dir(config: TriggerModeBenchmarkConfig) -> Path:
    tag = config.run_tag
    if not tag:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_base_dir).expanduser() / str(tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def write_config(run_dir: Path, config: TriggerModeBenchmarkConfig) -> None:
    payload = asdict(config)
    payload["output_base_dir"] = str(payload["output_base_dir"])
    payload["mode_names"] = list(payload["mode_names"])
    payload["posterior_probability_thresholds"] = list(payload["posterior_probability_thresholds"])
    payload["logbf_thresholds"] = list(payload["logbf_thresholds"])
    with (run_dir / "config.json").open("w", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _threshold_token(value: float) -> str:
    text = f"{float(value):.8g}"
    return text.replace("-", "m").replace(".", "p")


def _profile_label(family: str, threshold: float, is_production: bool) -> str:
    if family == "loo_posterior_prob":
        text = f"LOO posterior P(event) >= {threshold:.8g}"
    else:
        text = f"local log BF >= {threshold:.8g}"
    if is_production:
        text += " (production)"
    return text


def trigger_profiles(config: TriggerModeBenchmarkConfig) -> list[dict[str, object]]:
    profiles: list[dict[str, object]] = []
    production_prob = posterior_probability_threshold(config.significance_threshold)
    seen_prob: set[float] = set()
    for threshold in config.posterior_probability_thresholds:
        prob = posterior_probability_threshold(float(threshold))
        key = round(prob, 12)
        if key in seen_prob:
            continue
        seen_prob.add(key)
        is_production = np.isclose(prob, production_prob, rtol=0.0, atol=1e-12)
        name = f"loo_p_ge_{_threshold_token(prob)}"
        if is_production:
            name += "_production"
        profiles.append(
            {
                "trigger_profile": name,
                "trigger_profile_label": _profile_label("loo_posterior_prob", prob, is_production),
                "trigger_family": "loo_posterior_prob",
                "trigger_mode": "posterior_prob",
                "threshold": float(prob),
                "is_production": bool(is_production),
            }
        )

    seen_bf: set[float] = set()
    for threshold in config.logbf_thresholds:
        logbf = float(threshold)
        key = round(logbf, 12)
        if key in seen_bf:
            continue
        seen_bf.add(key)
        is_production = np.isclose(logbf, float(config.logbf_threshold_dip), rtol=0.0, atol=1e-12)
        name = f"logbf_ge_{_threshold_token(logbf)}"
        if is_production:
            name += "_production"
        profiles.append(
            {
                "trigger_profile": name,
                "trigger_profile_label": _profile_label("local_logbf", logbf, is_production),
                "trigger_family": "local_logbf",
                "trigger_mode": "logbf",
                "threshold": float(logbf),
                "is_production": bool(is_production),
            }
        )
    return profiles


def _baseline_kwargs_for_mode(row: pd.Series, mode_name: str, config_dict: dict[str, object]) -> tuple[object, dict[str, object], float]:
    selected_period = _mode_selected_period(row, mode_name)
    baseline_kwargs: dict[str, object] = {
        "S0": float(config_dict["baseline_s0"]),
        "w0": float(config_dict["baseline_w0"]),
        "q": float(config_dict["baseline_q"]),
        "jitter": float(config_dict["baseline_jitter"]),
        "sigma_floor": config_dict["baseline_sigma_floor"],
        "add_sigma_eff_col": True,
    }
    spec = PERIODIC_BRANCH_MODE_SPECS[mode_name]
    if spec["baseline"] == "phase_template":
        baseline_kwargs["period_days"] = selected_period
        return phase_template_baseline, baseline_kwargs, selected_period
    if spec["baseline"] == "gp_masked":
        return per_camera_gp_baseline_masked, baseline_kwargs, selected_period
    raise ValueError(f"Unsupported mode: {mode_name}")


def _run_profile(
    profile: dict[str, object],
    *,
    jd: np.ndarray,
    cam_vec: np.ndarray | None,
    log_bf_local: np.ndarray,
    event_probability: np.ndarray,
    config_dict: dict[str, object],
) -> dict[str, object]:
    trigger_mode = str(profile["trigger_mode"])
    threshold = float(profile["threshold"])
    trigger = resolve_trigger_indices(
        trigger_mode=trigger_mode,
        log_bf_local=log_bf_local,
        event_probability=event_probability,
        logbf_threshold=threshold if trigger_mode == "logbf" else float(config_dict["logbf_threshold_dip"]),
        significance_threshold=threshold if trigger_mode == "posterior_prob" else float(config_dict["significance_threshold"]),
    )
    point_significance = np.asarray(trigger["point_significance"], dtype=float)
    raw_idx = np.asarray(trigger["event_indices"], dtype=int)

    runs = build_runs(
        raw_idx,
        jd,
        max_gap_points=int(config_dict["run_max_gap_points"]),
        max_gap_days=config_dict["run_max_gap_days"],
    )
    kept_runs, _ = filter_runs(
        runs,
        jd,
        point_significance,
        min_points=int(config_dict["run_min_points"]),
        min_duration_days=config_dict["run_min_duration_days"],
        per_point_threshold=float(trigger["trigger_threshold"]),
        cam_vec=cam_vec,
    )
    if kept_runs:
        event_indices = np.unique(np.concatenate(kept_runs)).astype(int)
        significant = True
    else:
        event_indices = np.array([], dtype=int)
        significant = False
    run_stats = summarize_kept_runs(kept_runs, jd, point_significance, cam_vec=cam_vec)
    return {
        "event_indices": event_indices,
        "raw_trigger_points": int(raw_idx.size),
        "dip_significant": bool(significant),
        "trigger_max": float(trigger["trigger_max"]) if np.isfinite(trigger["trigger_max"]) else np.nan,
        "trigger_threshold": float(trigger["trigger_threshold"]),
        **run_stats,
    }


def _base_score_record(
    *,
    row: pd.Series,
    mode_name: str,
    selected_period: float,
    clean_df: pd.DataFrame,
    df_base: pd.DataFrame,
    dip: dict[str, object],
) -> dict[str, object]:
    truth_mask = clean_df["truth_dip_mask"].to_numpy(dtype=bool)
    has_dip = bool(row["has_dip"])
    truth_observable = bool(clean_df["truth_observable"].iloc[0]) if len(clean_df) else False
    outside = ~truth_mask
    if not outside.any():
        outside = np.ones(len(clean_df), dtype=bool)

    baseline_values = pd.to_numeric(df_base["baseline"], errors="coerce").to_numpy(dtype=float)
    true_baseline = pd.to_numeric(clean_df["true_baseline_mag"], errors="coerce").to_numpy(dtype=float)
    baseline_error = baseline_values - true_baseline
    resid = pd.to_numeric(df_base["resid"], errors="coerce").to_numpy(dtype=float)
    truth_signal = pd.to_numeric(clean_df["truth_dip_signal"], errors="coerce").to_numpy(dtype=float)
    errors = pd.to_numeric(clean_df["error"], errors="coerce").to_numpy(dtype=float)
    log_bf_local = np.asarray(dip.get("log_bf_local", []), dtype=float)
    event_probability = np.asarray(dip.get("event_probability", []), dtype=float)

    recovered_amp = float(np.nanmax(resid[truth_mask])) if truth_mask.any() and np.isfinite(resid[truth_mask]).any() else np.nan
    true_amp_sampled = float(np.nanmax(truth_signal)) if np.isfinite(truth_signal).any() else np.nan
    baseline_source = str(dip.get("baseline_source", "unknown"))

    if truth_mask.any():
        max_prob_truth = float(np.nanmax(event_probability[truth_mask])) if np.isfinite(event_probability[truth_mask]).any() else np.nan
        max_logbf_truth = float(np.nanmax(log_bf_local[truth_mask])) if np.isfinite(log_bf_local[truth_mask]).any() else np.nan
    else:
        max_prob_truth = np.nan
        max_logbf_truth = np.nan

    if outside.any():
        max_prob_outside = float(np.nanmax(event_probability[outside])) if np.isfinite(event_probability[outside]).any() else np.nan
        max_logbf_outside = float(np.nanmax(log_bf_local[outside])) if np.isfinite(log_bf_local[outside]).any() else np.nan
    else:
        max_prob_outside = np.nan
        max_logbf_outside = np.nan

    return {
        "trial_id": int(row["trial_id"]),
        "mode": mode_name,
        "mode_label": str(PERIODIC_BRANCH_MODE_SPECS[mode_name]["label"]),
        "status": "ok",
        "error": "",
        "selected_period_days": selected_period,
        "period_frac_error": (
            abs(selected_period - float(row["period_days"])) / float(row["period_days"])
            if np.isfinite(selected_period)
            else np.nan
        ),
        "n_points_actual": int(len(clean_df)),
        "n_cameras_actual": int(clean_df["camera#"].nunique()) if "camera#" in clean_df.columns else 0,
        "truth_support_points_actual": int(np.count_nonzero(truth_mask)),
        "truth_peak_snr_actual": float(np.nanmax(truth_signal / errors)) if has_dip and len(clean_df) and np.isfinite(truth_signal / errors).any() else 0.0,
        "truth_observable_actual": truth_observable,
        "bayes_factor": float(dip.get("bayes_factor", np.nan)),
        "max_log_bf_local": float(dip.get("max_log_bf_local", np.nan)),
        "max_event_probability": float(np.nanmax(event_probability)) if event_probability.size and np.isfinite(event_probability).any() else np.nan,
        "max_event_probability_truth": max_prob_truth,
        "max_log_bf_local_truth": max_logbf_truth,
        "max_event_probability_outside_truth": max_prob_outside,
        "max_log_bf_local_outside_truth": max_logbf_outside,
        "baseline_source": baseline_source,
        "phase_template_fallback": bool("phase_template_fallback" in baseline_source),
        "baseline_mae_outside_dip": float(np.nanmedian(np.abs(baseline_error[outside]))),
        "baseline_rmse_outside_dip": float(np.sqrt(np.nanmean(baseline_error[outside] ** 2))),
        "resid_rms_outside_dip": float(np.sqrt(np.nanmean(resid[outside] ** 2))),
        "resid_mad_outside_dip": float(1.4826 * np.nanmedian(np.abs(resid[outside] - np.nanmedian(resid[outside])))),
        "recovered_amp_mag": recovered_amp,
        "true_amp_sampled_mag": true_amp_sampled,
        "amp_recovery_ratio": float(recovered_amp / true_amp_sampled) if np.isfinite(recovered_amp) and true_amp_sampled > 0 else np.nan,
    }


def _profile_detection_record(
    *,
    base: dict[str, object],
    profile: dict[str, object],
    profile_result: dict[str, object],
    clean_df: pd.DataFrame,
) -> dict[str, object]:
    event_indices = np.asarray(profile_result["event_indices"], dtype=int)
    event_indices = event_indices[(event_indices >= 0) & (event_indices < len(clean_df))]
    event_mask = np.zeros(len(clean_df), dtype=bool)
    event_mask[event_indices] = True
    truth_mask = clean_df["truth_dip_mask"].to_numpy(dtype=bool)
    has_dip = bool(clean_df["dip_class"].iloc[0] != "control_none") if len(clean_df) else False
    overlap_points = int(np.count_nonzero(event_mask & truth_mask))
    detected_overlap = bool(overlap_points > 0)
    dip_significant = bool(profile_result["dip_significant"])
    target_recovered = bool(has_dip and dip_significant and detected_overlap)
    false_positive = bool((not has_dip) and dip_significant)
    off_target_detection = bool(has_dip and dip_significant and not detected_overlap)

    return {
        **base,
        "trigger_profile": str(profile["trigger_profile"]),
        "trigger_profile_label": str(profile["trigger_profile_label"]),
        "trigger_family": str(profile["trigger_family"]),
        "trigger_mode": str(profile["trigger_mode"]),
        "threshold": float(profile["threshold"]),
        "is_production": bool(profile["is_production"]),
        "dip_significant": dip_significant,
        "target_recovered": target_recovered,
        "false_positive": false_positive,
        "off_target_detection": off_target_detection,
        "detected_overlap": detected_overlap,
        "overlap_points": overlap_points,
        "event_points": int(event_mask.sum()),
        "raw_trigger_points": int(profile_result["raw_trigger_points"]),
        "trigger_max": float(profile_result["trigger_max"]) if np.isfinite(profile_result["trigger_max"]) else np.nan,
        "trigger_threshold": float(profile_result["trigger_threshold"]),
        "dip_run_count": int(profile_result["n_runs"]),
        "dip_max_run_points": int(profile_result["max_run_points"]),
        "dip_max_run_duration": float(profile_result["max_run_duration"]) if np.isfinite(profile_result["max_run_duration"]) else np.nan,
        "dip_max_run_sum": float(profile_result["max_run_sum"]) if np.isfinite(profile_result["max_run_sum"]) else np.nan,
        "dip_max_run_max": float(profile_result["max_run_max"]) if np.isfinite(profile_result["max_run_max"]) else np.nan,
        "dip_max_run_cameras": int(profile_result["max_run_cameras"]),
    }


def _evaluate_task(task: tuple[dict[str, object], str, dict[str, object], list[dict[str, object]]]) -> dict[str, object]:
    row_dict, mode_name, config_dict, profiles = task
    row = pd.Series(row_dict)
    try:
        df = simulate_periodic_lightcurve(row)
        clean_df = clean_lc(df).reset_index(drop=True)
        baseline_func, baseline_kwargs, selected_period = _baseline_kwargs_for_mode(row, mode_name, config_dict)
        df_base = baseline_func(clean_df, **baseline_kwargs).reset_index(drop=True)
        dip = score_events_bayesian(
            clean_df,
            kind="dip",
            baseline_func=None,
            df_base=df_base,
            p_points=int(config_dict["p_points"]),
            mag_points=int(config_dict["mag_points"]),
            trigger_mode="posterior_prob",
            logbf_threshold=float(config_dict["logbf_threshold_dip"]),
            significance_threshold=float(config_dict["significance_threshold"]),
            run_min_points=int(config_dict["run_min_points"]),
            max_gap_points=int(config_dict["run_max_gap_points"]),
            run_max_gap_days=config_dict["run_max_gap_days"],
            run_min_duration_days=config_dict["run_min_duration_days"],
            compute_event_prob=True,
        )

        log_bf_local = np.asarray(dip.get("log_bf_local", []), dtype=float)
        event_probability = np.asarray(dip.get("event_probability", []), dtype=float)
        jd = clean_df["JD"].to_numpy(dtype=float)
        cam_vec = clean_df["camera#"].to_numpy() if "camera#" in clean_df.columns else None
        base = _base_score_record(
            row=row,
            mode_name=mode_name,
            selected_period=selected_period,
            clean_df=clean_df,
            df_base=df_base,
            dip=dip,
        )

        profile_rows: list[dict[str, object]] = []
        for profile in profiles:
            profile_result = _run_profile(
                profile,
                jd=jd,
                cam_vec=cam_vec,
                log_bf_local=log_bf_local,
                event_probability=event_probability,
                config_dict=config_dict,
            )
            profile_rows.append(
                _profile_detection_record(
                    base=base,
                    profile=profile,
                    profile_result=profile_result,
                    clean_df=clean_df,
                )
            )
        return {"score": base, "profiles": profile_rows}
    except Exception as exc:
        base = {
            "trial_id": int(row_dict.get("trial_id", -1)),
            "mode": mode_name,
            "mode_label": str(PERIODIC_BRANCH_MODE_SPECS.get(mode_name, {}).get("label", mode_name)),
            "status": "error",
            "error": str(exc),
        }
        return {
            "score": base,
            "profiles": [
                {
                    **base,
                    "trigger_profile": str(profile["trigger_profile"]),
                    "trigger_profile_label": str(profile["trigger_profile_label"]),
                    "trigger_family": str(profile["trigger_family"]),
                    "trigger_mode": str(profile["trigger_mode"]),
                    "threshold": float(profile["threshold"]),
                    "is_production": bool(profile["is_production"]),
                }
                for profile in profiles
            ],
        }


def _config_to_worker_dict(config: TriggerModeBenchmarkConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["output_base_dir"] = str(payload["output_base_dir"])
    payload["mode_names"] = list(payload["mode_names"])
    payload["posterior_probability_thresholds"] = list(payload["posterior_probability_thresholds"])
    payload["logbf_thresholds"] = list(payload["logbf_thresholds"])
    return payload


def evaluate_design(
    design: pd.DataFrame,
    config: TriggerModeBenchmarkConfig,
    *,
    score_output_path: Path | None = None,
    trigger_output_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles = trigger_profiles(config)
    cfg = _config_to_worker_dict(config)
    tasks: list[tuple[dict[str, object], str, dict[str, object], list[dict[str, object]]]] = []
    for _, row in design.iterrows():
        row_dict = row.to_dict()
        for mode_name in config.mode_names:
            if mode_name not in PERIODIC_BRANCH_MODE_SPECS:
                raise ValueError(f"Unknown benchmark mode: {mode_name}")
            tasks.append((row_dict, str(mode_name), cfg, profiles))

    score_rows: list[dict[str, object]] = []
    trigger_rows: list[dict[str, object]] = []
    if config.workers and int(config.workers) > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(config.workers)) as executor:
            futures = [executor.submit(_evaluate_task, task) for task in tasks]
            iterator = as_completed(futures)
            if config.show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="Trigger-mode simulations")
            for future in iterator:
                result = future.result()
                score_rows.append(result["score"])
                trigger_rows.extend(result["profiles"])
    else:
        iterator = tasks
        if config.show_progress:
            iterator = tqdm(tasks, desc="Trigger-mode simulations")
        for task in iterator:
            result = _evaluate_task(task)
            score_rows.append(result["score"])
            trigger_rows.extend(result["profiles"])

    score_df = pd.DataFrame(score_rows)
    trigger_df = pd.DataFrame(trigger_rows)
    if score_output_path is not None:
        score_output_path.parent.mkdir(parents=True, exist_ok=True)
        score_df.to_parquet(score_output_path, index=False, compression=PARQUET_OUTPUT_COMPRESSION)
    if trigger_output_path is not None:
        trigger_output_path.parent.mkdir(parents=True, exist_ok=True)
        trigger_df.to_parquet(trigger_output_path, index=False, compression=PARQUET_OUTPUT_COMPRESSION)
    return score_df, trigger_df


def merge_design_results(design: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    merged = results.merge(design, on="trial_id", how="left", suffixes=("", "_design"))
    return add_metric_bins(merged)


def _rate(series: pd.Series) -> float:
    if len(series) == 0:
        return np.nan
    return float(series.fillna(False).astype(bool).mean())


def summarize_trigger_results(df: pd.DataFrame, group_cols: list[str] | tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.groupby(list(group_cols), dropna=False) if group_cols else [((), df)]
    for key, sub_all in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {col: val for col, val in zip(group_cols, key_tuple)}
        ok = sub_all[sub_all["status"].eq("ok")].copy()
        has_dip = ok["has_dip"].fillna(False).astype(bool) if "has_dip" in ok.columns else pd.Series(dtype=bool)
        observable = has_dip & ok["truth_observable_actual"].fillna(False).astype(bool) if len(ok) else pd.Series(dtype=bool)
        controls = ~has_dip if len(ok) else pd.Series(dtype=bool)
        detected = ok["dip_significant"].fillna(False).astype(bool) if "dip_significant" in ok.columns else pd.Series(dtype=bool)
        recovered = ok["target_recovered"].fillna(False).astype(bool) if "target_recovered" in ok.columns else pd.Series(dtype=bool)
        true_positive_detected = detected & has_dip & ok["detected_overlap"].fillna(False).astype(bool) if len(ok) else pd.Series(dtype=bool)

        row.update(
            {
                "n": int(len(sub_all)),
                "n_ok": int(len(ok)),
                "n_dip": int(has_dip.sum()) if len(ok) else 0,
                "n_observable_dip": int(observable.sum()) if len(ok) else 0,
                "n_control": int(controls.sum()) if len(ok) else 0,
                "status_error_rate": float(1.0 - len(ok) / max(1, len(sub_all))),
                "dip_significant_rate": _rate(detected),
                "observable_recall": float(recovered[observable].mean()) if len(ok) and observable.any() else np.nan,
                "all_dip_recall": float(recovered[has_dip].mean()) if len(ok) and has_dip.any() else np.nan,
                "control_false_positive_rate": float(detected[controls].mean()) if len(ok) and controls.any() else np.nan,
                "off_target_detection_rate": _rate(ok.loc[has_dip, "off_target_detection"]) if len(ok) and has_dip.any() else np.nan,
                "precision_by_trial": float(true_positive_detected.sum() / detected.sum()) if len(ok) and detected.any() else np.nan,
                "phase_template_fallback_rate": _rate(ok["phase_template_fallback"]) if "phase_template_fallback" in ok.columns else np.nan,
                "median_raw_trigger_points": float(ok["raw_trigger_points"].median()) if "raw_trigger_points" in ok.columns and len(ok) else np.nan,
                "median_event_points": float(ok["event_points"].median()) if "event_points" in ok.columns and len(ok) else np.nan,
                "median_trigger_max": float(ok["trigger_max"].median()) if "trigger_max" in ok.columns and len(ok) else np.nan,
                "median_max_event_probability": float(ok["max_event_probability"].median()) if "max_event_probability" in ok.columns and len(ok) else np.nan,
                "median_max_log_bf_local": float(ok["max_log_bf_local"].median()) if "max_log_bf_local" in ok.columns and len(ok) else np.nan,
                "median_bayes_factor": float(ok["bayes_factor"].median()) if "bayes_factor" in ok.columns and len(ok) else np.nan,
                "median_baseline_mae_outside_dip": float(ok["baseline_mae_outside_dip"].median()) if "baseline_mae_outside_dip" in ok.columns and len(ok) else np.nan,
                "median_resid_rms_outside_dip": float(ok["resid_rms_outside_dip"].median()) if "resid_rms_outside_dip" in ok.columns and len(ok) else np.nan,
                "median_amp_recovery_ratio": float(ok.loc[has_dip, "amp_recovery_ratio"].median()) if len(ok) and has_dip.any() else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_pairwise_production_comparison(df: pd.DataFrame) -> pd.DataFrame:
    prod = df[df["is_production"].fillna(False).astype(bool) & df["status"].eq("ok")].copy()
    rows: list[dict[str, object]] = []
    for mode, sub in prod.groupby("mode", dropna=False):
        posterior = sub[sub["trigger_family"].eq("loo_posterior_prob")].set_index("trial_id")
        logbf = sub[sub["trigger_family"].eq("local_logbf")].set_index("trial_id")
        common = posterior.index.intersection(logbf.index)
        if common.empty:
            continue
        pp = posterior.loc[common]
        bf = logbf.loc[common]
        has_dip = pp["has_dip"].fillna(False).astype(bool)
        observable = has_dip & pp["truth_observable_actual"].fillna(False).astype(bool)
        controls = ~has_dip

        pp_sig = pp["dip_significant"].fillna(False).astype(bool)
        bf_sig = bf["dip_significant"].fillna(False).astype(bool)
        pp_rec = pp["target_recovered"].fillna(False).astype(bool)
        bf_rec = bf["target_recovered"].fillna(False).astype(bool)
        pp_fp = pp["false_positive"].fillna(False).astype(bool)
        bf_fp = bf["false_positive"].fillna(False).astype(bool)

        rows.append(
            {
                "mode": mode,
                "mode_label": pp["mode_label"].iloc[0],
                "n_common": int(len(common)),
                "n_observable_dip": int(observable.sum()),
                "n_control": int(controls.sum()),
                "posterior_observable_recall": float(pp_rec[observable].mean()) if observable.any() else np.nan,
                "logbf_observable_recall": float(bf_rec[observable].mean()) if observable.any() else np.nan,
                "both_recovered_rate": float((pp_rec & bf_rec)[observable].mean()) if observable.any() else np.nan,
                "posterior_only_recovered_rate": float((pp_rec & ~bf_rec)[observable].mean()) if observable.any() else np.nan,
                "logbf_only_recovered_rate": float((~pp_rec & bf_rec)[observable].mean()) if observable.any() else np.nan,
                "both_missed_rate": float((~pp_rec & ~bf_rec)[observable].mean()) if observable.any() else np.nan,
                "posterior_control_false_positive_rate": float(pp_fp[controls].mean()) if controls.any() else np.nan,
                "logbf_control_false_positive_rate": float(bf_fp[controls].mean()) if controls.any() else np.nan,
                "both_false_positive_rate": float((pp_fp & bf_fp)[controls].mean()) if controls.any() else np.nan,
                "posterior_only_false_positive_rate": float((pp_fp & ~bf_fp)[controls].mean()) if controls.any() else np.nan,
                "logbf_only_false_positive_rate": float((~pp_fp & bf_fp)[controls].mean()) if controls.any() else np.nan,
                "detection_agreement_rate": float((pp_sig == bf_sig).mean()),
                "recovery_agreement_rate_observable": float((pp_rec[observable] == bf_rec[observable]).mean()) if observable.any() else np.nan,
                "median_event_points_delta_posterior_minus_logbf": float((pp["event_points"] - bf["event_points"]).median()),
            }
        )
    return pd.DataFrame(rows)


def build_summary_slices(merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    production = merged[merged["is_production"].fillna(False).astype(bool)].copy()
    return {
        "all_thresholds_overall": summarize_trigger_results(
            merged,
            ["mode", "trigger_family", "trigger_profile", "threshold"],
        ),
        "production_overall": summarize_trigger_results(
            production,
            ["mode", "trigger_family", "trigger_profile", "threshold"],
        ),
        "production_by_amp": summarize_trigger_results(
            production,
            ["mode", "trigger_family", "dip_amp_bin"],
        ),
        "production_by_period": summarize_trigger_results(
            production,
            ["mode", "trigger_family", "period_bin"],
        ),
        "production_by_width": summarize_trigger_results(
            production,
            ["mode", "trigger_family", "width_bin"],
        ),
        "production_by_points": summarize_trigger_results(
            production,
            ["mode", "trigger_family", "points_bin_actual"],
        ),
        "production_by_waveform": summarize_trigger_results(
            production,
            ["mode", "trigger_family", "waveform_kind"],
        ),
    }


def run_trigger_mode_simulation_benchmark(config: TriggerModeBenchmarkConfig) -> TriggerModeBenchmarkRun:
    run_dir = make_run_dir(config)
    write_config(run_dir, config)

    design_path = run_dir / "trial_design.parquet"
    score_path = run_dir / "score_results.parquet"
    trigger_path = run_dir / "trigger_results.parquet"

    if design_path.exists() and not config.force:
        design = pd.read_parquet(design_path)
    else:
        design = generate_trial_design(config)
        design.to_parquet(design_path, index=False, compression=PARQUET_OUTPUT_COMPRESSION)

    if score_path.exists() and trigger_path.exists() and not config.force:
        score_results = pd.read_parquet(score_path)
        trigger_results = pd.read_parquet(trigger_path)
    else:
        score_results, trigger_results = evaluate_design(
            design,
            config,
            score_output_path=score_path,
            trigger_output_path=trigger_path,
        )

    score_merged = merge_design_results(design, score_results)
    trigger_merged = merge_design_results(design, trigger_results)
    score_merged.to_parquet(run_dir / "score_results_with_design.parquet", index=False, compression=PARQUET_OUTPUT_COMPRESSION)
    trigger_merged.to_parquet(run_dir / "trigger_results_with_design.parquet", index=False, compression=PARQUET_OUTPUT_COMPRESSION)

    summary_slices = build_summary_slices(trigger_merged)
    pairwise = build_pairwise_production_comparison(trigger_merged)
    summary_dir = run_dir / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for name, table in summary_slices.items():
        table.to_parquet(summary_dir / f"{name}.parquet", index=False, compression=PARQUET_OUTPUT_COMPRESSION)
        table.to_csv(summary_dir / f"{name}.csv", index=False)
    pairwise.to_parquet(summary_dir / "pairwise_production.parquet", index=False, compression=PARQUET_OUTPUT_COMPRESSION)
    pairwise.to_csv(summary_dir / "pairwise_production.csv", index=False)

    return TriggerModeBenchmarkRun(
        config=config,
        run_dir=run_dir,
        trial_design=design,
        score_results=score_merged,
        trigger_results=trigger_merged,
        summary_slices=summary_slices,
        pairwise_production=pairwise,
    )


def load_trigger_mode_simulation_benchmark(run_dir: Path | str) -> TriggerModeBenchmarkRun:
    run_dir = Path(run_dir).expanduser()
    with (run_dir / "config.json").open("r", encoding="ascii") as handle:
        cfg_payload = json.load(handle)
    cfg_payload["output_base_dir"] = Path(cfg_payload["output_base_dir"])
    cfg_payload["mode_names"] = tuple(cfg_payload["mode_names"])
    cfg_payload["posterior_probability_thresholds"] = tuple(cfg_payload["posterior_probability_thresholds"])
    cfg_payload["logbf_thresholds"] = tuple(cfg_payload["logbf_thresholds"])
    config = TriggerModeBenchmarkConfig(**cfg_payload)
    design = pd.read_parquet(run_dir / "trial_design.parquet")
    score_results = pd.read_parquet(run_dir / "score_results_with_design.parquet")
    trigger_results = pd.read_parquet(run_dir / "trigger_results_with_design.parquet")
    summary_slices = build_summary_slices(trigger_results)
    pairwise = build_pairwise_production_comparison(trigger_results)
    return TriggerModeBenchmarkRun(
        config=config,
        run_dir=run_dir,
        trial_design=design,
        score_results=score_results,
        trigger_results=trigger_results,
        summary_slices=summary_slices,
        pairwise_production=pairwise,
    )


def recompute_trial_scores(
    run: TriggerModeBenchmarkRun,
    trial_id: int,
    *,
    mode: str = "phase_template_true_period",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    row = run.trial_design.loc[run.trial_design["trial_id"] == int(trial_id)]
    if row.empty:
        raise KeyError(f"trial_id {trial_id} not found")
    row_series = row.iloc[0]
    df = clean_lc(simulate_periodic_lightcurve(row_series)).reset_index(drop=True)
    baseline_func, baseline_kwargs, selected_period = _baseline_kwargs_for_mode(row_series, mode, _config_to_worker_dict(run.config))
    df_base = baseline_func(df, **baseline_kwargs).reset_index(drop=True)
    dip = score_events_bayesian(
        df,
        kind="dip",
        baseline_func=None,
        df_base=df_base,
        p_points=run.config.p_points,
        mag_points=run.config.mag_points,
        trigger_mode="posterior_prob",
        logbf_threshold=run.config.logbf_threshold_dip,
        significance_threshold=run.config.significance_threshold,
        run_min_points=run.config.run_min_points,
        max_gap_points=run.config.run_max_gap_points,
        run_max_gap_days=run.config.run_max_gap_days,
        run_min_duration_days=run.config.run_min_duration_days,
        compute_event_prob=True,
    )
    info = {
        "mode": mode,
        "mode_label": PERIODIC_BRANCH_MODE_SPECS[mode]["label"],
        "selected_period_days": selected_period,
        "dip": dip,
    }
    return df, df_base, info


def production_event_indices_for_trial(
    run: TriggerModeBenchmarkRun,
    df: pd.DataFrame,
    info: dict[str, object],
) -> dict[str, np.ndarray]:
    jd = df["JD"].to_numpy(dtype=float)
    cam_vec = df["camera#"].to_numpy() if "camera#" in df.columns else None
    dip = info["dip"]
    log_bf_local = np.asarray(dip.get("log_bf_local", []), dtype=float)
    event_probability = np.asarray(dip.get("event_probability", []), dtype=float)
    cfg = _config_to_worker_dict(run.config)
    out: dict[str, np.ndarray] = {}
    for profile in trigger_profiles(run.config):
        if not bool(profile["is_production"]):
            continue
        result = _run_profile(
            profile,
            jd=jd,
            cam_vec=cam_vec,
            log_bf_local=log_bf_local,
            event_probability=event_probability,
            config_dict=cfg,
        )
        out[str(profile["trigger_family"])] = np.asarray(result["event_indices"], dtype=int)
    return out


def plot_trigger_mode_trial_diagnostic(
    run: TriggerModeBenchmarkRun,
    trial_id: int,
    *,
    mode: str = "phase_template_true_period",
    ax: np.ndarray | None = None,
) -> np.ndarray:
    df, df_base, info = recompute_trial_scores(run, trial_id, mode=mode)
    if ax is None:
        _, ax = plt.subplots(5, 1, figsize=figsize_two_col_grid(1, 5), sharex=False)

    jd = df["JD"].to_numpy(dtype=float)
    truth_mask = df["truth_dip_mask"].to_numpy(dtype=bool)
    event_indices = production_event_indices_for_trial(run, df, info)
    posterior_idx = event_indices.get("loo_posterior_prob", np.array([], dtype=int))
    logbf_idx = event_indices.get("local_logbf", np.array([], dtype=int))
    dip = info["dip"]
    event_probability = np.asarray(dip.get("event_probability", []), dtype=float)
    log_bf_local = np.asarray(dip.get("log_bf_local", []), dtype=float)
    phase = np.mod((jd - np.nanmin(jd)) / float(df["period_days"].iloc[0]), 1.0)

    baseline_overlay = pd.DataFrame({"JD": jd, "baseline": df_base["baseline"].to_numpy(dtype=float)})
    plot_lightcurve_panel(
        ax[0],
        df,
        group_by="camera",
        camera_col="camera#",
        show_errorbars=False,
        marker_size=3.4,
        legend="none",
        time_offset="none",
        xlabel="JD",
        ylabel="mag",
        baseline=baseline_overlay,
        baseline_col="baseline",
        baseline_time_col="JD",
        baseline_label=str(info["mode_label"]),
        baseline_style={"color": "crimson", "linewidth": 1.1},
    )
    ax[0].plot(jd, df["true_baseline_mag"], color="black", lw=1.0, label="true baseline")
    if truth_mask.any():
        ax[0].scatter(jd[truth_mask], df.loc[truth_mask, "mag"], s=34, facecolors="none", edgecolors="limegreen", label="truth dip support")
    if posterior_idx.size:
        ax[0].scatter(jd[posterior_idx], df.loc[posterior_idx, "mag"], marker="x", s=46, color="gold", label="LOO posterior production")
    if logbf_idx.size:
        ax[0].scatter(jd[logbf_idx], df.loc[logbf_idx, "mag"], marker="+", s=56, color="tab:purple", label="logBF production")
    ax[0].set_title(f"Trial {trial_id}: observed light curve and baseline")
    ax[0].legend(ncol=4, fontsize=8)

    residual_plot = pd.DataFrame({"JD": jd, "resid": df_base["resid"].to_numpy(dtype=float)})
    plot_residual_panel(
        ax[1],
        residual_plot,
        group_by="none",
        show_errorbars=False,
        marker_size=3.4,
        legend="none",
        time_offset="none",
        xlabel="JD",
        ylabel="mag - baseline",
        invert_y=False,
    )
    if truth_mask.any():
        ax[1].scatter(jd[truth_mask], df_base.loc[truth_mask, "resid"], s=34, facecolors="none", edgecolors="limegreen")
    if posterior_idx.size:
        ax[1].scatter(jd[posterior_idx], df_base.loc[posterior_idx, "resid"], marker="x", s=46, color="gold")
    if logbf_idx.size:
        ax[1].scatter(jd[logbf_idx], df_base.loc[logbf_idx, "resid"], marker="+", s=56, color="tab:purple")
    ax[1].set_title("Residuals used by Bayesian event scoring")

    ax[2].scatter(jd, event_probability, s=12, alpha=0.55)
    ax[2].axhline(posterior_probability_threshold(run.config.significance_threshold), color="goldenrod", lw=1.0, label="production threshold")
    ax[2].set_ylim(-0.02, 1.02)
    ax[2].set_ylabel("LOO P(event)")
    ax[2].set_title("LOO posterior probability by point")
    ax[2].legend(fontsize=8)

    ax[3].scatter(jd, log_bf_local, s=12, alpha=0.55)
    ax[3].axhline(run.config.logbf_threshold_dip, color="tab:purple", lw=1.0, label="production threshold")
    ax[3].set_ylabel("local log BF")
    ax[3].set_title("Local Bayes factor evidence by point")
    ax[3].legend(fontsize=8)
    style_publication_axis(ax[2])
    style_publication_axis(ax[3])

    plot_phase_panel(
        ax[4],
        df,
        period_days=float(df["period_days"].iloc[0]),
        epoch_jd=float(np.nanmin(jd)),
        group_by="none",
        show_errorbars=False,
        marker_size=3.4,
        legend="none",
    )
    order = np.argsort(phase)
    baseline_values = df_base["baseline"].to_numpy(dtype=float)
    ax[4].plot(phase[order], baseline_values[order], color="crimson", lw=1.0)
    ax[4].plot(phase[order] + 1.0, baseline_values[order], color="crimson", lw=1.0, alpha=0.7)
    ax[4].set_title("Folded view")
    return ax


def select_disagreement_trials(
    df: pd.DataFrame,
    *,
    mode: str = "phase_template_true_period",
) -> dict[str, int | None]:
    prod = df[df["is_production"].fillna(False).astype(bool) & df["mode"].eq(mode) & df["status"].eq("ok")].copy()
    posterior = prod[prod["trigger_family"].eq("loo_posterior_prob")].set_index("trial_id")
    logbf = prod[prod["trigger_family"].eq("local_logbf")].set_index("trial_id")
    common = posterior.index.intersection(logbf.index)
    examples: dict[str, int | None] = {
        "both_recovered": None,
        "posterior_only_recovered": None,
        "logbf_only_recovered": None,
        "posterior_only_false_positive": None,
        "logbf_only_false_positive": None,
        "both_false_positive": None,
    }
    if common.empty:
        return examples

    pp = posterior.loc[common]
    bf = logbf.loc[common]
    pp_rec = pp["target_recovered"].fillna(False).astype(bool)
    bf_rec = bf["target_recovered"].fillna(False).astype(bool)
    pp_fp = pp["false_positive"].fillna(False).astype(bool)
    bf_fp = bf["false_positive"].fillna(False).astype(bool)

    def pick(mask: pd.Series, sort_source: pd.DataFrame, sort_col: str, ascending: bool = False) -> int | None:
        idx = mask[mask].index
        if len(idx) == 0:
            return None
        return int(sort_source.loc[idx].sort_values(sort_col, ascending=ascending).index[0])

    examples["both_recovered"] = pick(pp_rec & bf_rec, pp, "max_event_probability", ascending=False)
    examples["posterior_only_recovered"] = pick(pp_rec & ~bf_rec, pp, "max_event_probability", ascending=False)
    examples["logbf_only_recovered"] = pick(~pp_rec & bf_rec, bf, "max_log_bf_local", ascending=False)
    examples["posterior_only_false_positive"] = pick(pp_fp & ~bf_fp, pp, "max_event_probability", ascending=False)
    examples["logbf_only_false_positive"] = pick(~pp_fp & bf_fp, bf, "max_log_bf_local", ascending=False)
    examples["both_false_positive"] = pick(pp_fp & bf_fp, pp, "max_event_probability", ascending=False)
    return examples
