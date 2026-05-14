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

from malca.baseline import per_camera_gp_baseline, per_camera_gp_baseline_masked
from malca.config import (
    BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    BASELINE_JITTER,
    BASELINE_Q,
    BASELINE_S0,
    BASELINE_W0,
    LOGBF_THRESHOLD_DIP,
    LOGBF_THRESHOLD_JUMP,
    MAG_POINTS,
    PARQUET_OUTPUT_COMPRESSION,
    P_POINTS,
    RUN_MAX_GAP_POINTS,
    RUN_MIN_POINTS,
    SIGNIFICANCE_THRESHOLD,
)
from malca.events import build_runs, filter_runs, score_lightcurve, summarize_kept_runs
from malca.triggering import posterior_probability_threshold, resolve_trigger_indices
from malca.utils import clean_lc, filter_bad_cameras, read_lc_dat2, read_skypatrol_csv


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
class March18FlatTriggerConfig:
    output_base_dir: Path = Path("output/diagnostics/march18_flat_trigger_mode_benchmark")
    run_tag: str | None = None
    flat_lc_dir: Path = Path("output/runs/runs_march18_bundle_all/bundle_assets/lightcurves")
    manifest_path: Path | None = Path("output/runs/local_lc_all_run/manifests/lc_manifest_all.parquet")
    index_file: Path | None = Path("output/runs/local_lc_all_magbin_index.parquet")
    extension: str = "dat3"
    max_sources: int | None = None
    seed: int = 20260514
    workers: int = 8
    show_progress: bool = True
    force: bool = False

    # Use the full per-camera GP by default for this real-light-curve run.
    baseline_func: str = "gp"
    auto_filter_bad_cameras: bool = True
    bad_camera_scatter_ratio: float = BAD_CAMERA_SCATTER_RATIO_THRESHOLD

    logbf_threshold_dip: float = LOGBF_THRESHOLD_DIP
    logbf_threshold_jump: float = LOGBF_THRESHOLD_JUMP
    significance_threshold: float = SIGNIFICANCE_THRESHOLD
    posterior_probability_thresholds: tuple[float, ...] = DEFAULT_POSTERIOR_PROBABILITY_THRESHOLDS
    logbf_thresholds: tuple[float, ...] = DEFAULT_LOGBF_THRESHOLDS
    event_kinds: tuple[str, ...] = ("dip", "jump")

    p_points: int = P_POINTS
    mag_points: int = MAG_POINTS
    run_min_points: int = RUN_MIN_POINTS
    run_max_gap_points: int = RUN_MAX_GAP_POINTS
    run_max_gap_days: float | None = None
    run_min_duration_days: float = 0.0

    baseline_s0: float = BASELINE_S0
    baseline_w0: float = BASELINE_W0
    baseline_q: float = BASELINE_Q
    baseline_jitter: float = BASELINE_JITTER
    baseline_sigma_floor: float | None = None


@dataclass
class March18FlatTriggerRun:
    config: March18FlatTriggerConfig
    run_dir: Path
    manifest: pd.DataFrame
    score_results: pd.DataFrame
    trigger_results: pd.DataFrame
    summary_overall: pd.DataFrame
    pairwise_production: pd.DataFrame


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _config_to_dict(config: March18FlatTriggerConfig) -> dict[str, object]:
    payload = asdict(config)
    for key in ("output_base_dir", "flat_lc_dir", "manifest_path", "index_file"):
        if payload.get(key) is not None:
            payload[key] = str(payload[key])
    payload["posterior_probability_thresholds"] = list(payload["posterior_probability_thresholds"])
    payload["logbf_thresholds"] = list(payload["logbf_thresholds"])
    payload["event_kinds"] = list(payload["event_kinds"])
    return payload


def make_run_dir(config: March18FlatTriggerConfig) -> Path:
    tag = config.run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.output_base_dir).expanduser() / str(tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_config(run_dir: Path, config: March18FlatTriggerConfig) -> None:
    with (run_dir / "config.json").open("w", encoding="ascii") as handle:
        json.dump(_config_to_dict(config), handle, indent=2, default=_json_default)


def load_config(run_dir: Path | str) -> March18FlatTriggerConfig:
    run_dir = Path(run_dir).expanduser()
    with (run_dir / "config.json").open("r", encoding="ascii") as handle:
        payload = json.load(handle)
    for key in ("output_base_dir", "flat_lc_dir", "manifest_path", "index_file"):
        if payload.get(key) is not None:
            payload[key] = Path(payload[key])
    payload["posterior_probability_thresholds"] = tuple(payload["posterior_probability_thresholds"])
    payload["logbf_thresholds"] = tuple(payload["logbf_thresholds"])
    payload["event_kinds"] = tuple(payload["event_kinds"])
    return March18FlatTriggerConfig(**payload)


def build_manifest(config: March18FlatTriggerConfig) -> pd.DataFrame:
    if config.manifest_path is not None and Path(config.manifest_path).expanduser().exists():
        manifest = pd.read_parquet(Path(config.manifest_path).expanduser())
    else:
        flat_lc_dir = Path(config.flat_lc_dir).expanduser()
        paths = sorted(flat_lc_dir.glob(f"*.{config.extension}"))
        manifest = pd.DataFrame(
            {
                "source_id": [p.stem for p in paths],
                "dat_path": [str(p) for p in paths],
            }
        )

    if "dat_path" not in manifest.columns:
        if "path" in manifest.columns:
            manifest = manifest.rename(columns={"path": "dat_path"})
        else:
            raise ValueError("Manifest must contain dat_path or path")

    manifest = manifest.copy()
    manifest["source_id"] = manifest["source_id"].astype(str) if "source_id" in manifest.columns else manifest["dat_path"].map(lambda x: Path(str(x)).stem)
    manifest["dat_path"] = manifest["dat_path"].astype(str)
    manifest = manifest[manifest["dat_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)

    if config.max_sources is not None and int(config.max_sources) < len(manifest):
        manifest = manifest.sample(n=int(config.max_sources), random_state=int(config.seed)).sort_values("source_id").reset_index(drop=True)
    return manifest


def _threshold_token(value: float) -> str:
    text = f"{float(value):.8g}"
    return text.replace("-", "m").replace(".", "p")


def trigger_profiles(config: March18FlatTriggerConfig, *, kind: str) -> list[dict[str, object]]:
    production_prob = posterior_probability_threshold(config.significance_threshold)
    logbf_production = config.logbf_threshold_jump if kind == "jump" else config.logbf_threshold_dip
    profiles: list[dict[str, object]] = []

    seen_prob: set[float] = set()
    for threshold in config.posterior_probability_thresholds:
        prob = posterior_probability_threshold(float(threshold))
        key = round(prob, 12)
        if key in seen_prob:
            continue
        seen_prob.add(key)
        is_production = np.isclose(prob, production_prob, rtol=0.0, atol=1e-12)
        name = f"{kind}_loo_p_ge_{_threshold_token(prob)}"
        if is_production:
            name += "_production"
        profiles.append(
            {
                "event_kind": kind,
                "trigger_profile": name,
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
        is_production = np.isclose(logbf, float(logbf_production), rtol=0.0, atol=1e-12)
        name = f"{kind}_logbf_ge_{_threshold_token(logbf)}"
        if is_production:
            name += "_production"
        profiles.append(
            {
                "event_kind": kind,
                "trigger_profile": name,
                "trigger_family": "local_logbf",
                "trigger_mode": "logbf",
                "threshold": float(logbf),
                "is_production": bool(is_production),
            }
        )
    return profiles


def _baseline_func_from_name(name: str):
    if str(name) == "gp":
        return per_camera_gp_baseline
    if str(name) == "gp_masked":
        return per_camera_gp_baseline_masked
    raise ValueError("This benchmark supports baseline_func='gp' or 'gp_masked'")


def _read_lightcurve(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return read_skypatrol_csv(str(path))
    source_id = path.stem
    file_ext = path.suffix[1:] if path.suffix else None
    df_g, df_v = read_lc_dat2(source_id, str(path.parent), file_ext=file_ext)
    if df_g.empty and df_v.empty:
        return pd.DataFrame()
    return pd.concat([df_g, df_v], ignore_index=True)


def _run_profile(
    profile: dict[str, object],
    *,
    jd: np.ndarray,
    cam_vec: np.ndarray | None,
    log_bf_local: np.ndarray,
    event_probability: np.ndarray,
    config_dict: dict[str, object],
    kind: str,
) -> dict[str, object]:
    trigger_mode = str(profile["trigger_mode"])
    threshold = float(profile["threshold"])
    logbf_default = float(config_dict["logbf_threshold_jump"] if kind == "jump" else config_dict["logbf_threshold_dip"])
    trigger = resolve_trigger_indices(
        trigger_mode=trigger_mode,
        log_bf_local=log_bf_local,
        event_probability=event_probability,
        logbf_threshold=threshold if trigger_mode == "logbf" else logbf_default,
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
    event_indices = np.unique(np.concatenate(kept_runs)).astype(int) if kept_runs else np.array([], dtype=int)
    run_stats = summarize_kept_runs(kept_runs, jd, point_significance, cam_vec=cam_vec)
    return {
        "event_indices": event_indices,
        "raw_trigger_points": int(raw_idx.size),
        "significant": bool(event_indices.size > 0),
        "trigger_max": float(trigger["trigger_max"]) if np.isfinite(trigger["trigger_max"]) else np.nan,
        "trigger_threshold": float(trigger["trigger_threshold"]),
        **run_stats,
    }


def _base_score_record(
    *,
    manifest_row: pd.Series,
    clean_df: pd.DataFrame,
    scored: dict[str, object],
    kind: str,
    pre_bad_cameras: set[object],
) -> dict[str, object]:
    branch = scored[kind]
    event_prob = np.asarray(branch.get("event_probability", []), dtype=float)
    log_bf = np.asarray(branch.get("log_bf_local", []), dtype=float)
    return {
        "source_id": str(manifest_row["source_id"]),
        "dat_path": str(manifest_row["dat_path"]),
        "mag_bin": str(manifest_row.get("mag_bin", "")),
        "event_kind": kind,
        "status": "ok",
        "error": "",
        "n_points": int(len(clean_df)),
        "n_cameras": int(clean_df["camera#"].nunique()) if "camera#" in clean_df.columns else 0,
        "pre_bad_camera_count": int(len(pre_bad_cameras)),
        "residual_bad_camera_count": int(len(scored.get("bad_cameras_filtered", set()))),
        "baseline_source": str(branch.get("baseline_source", "unknown")),
        "bayes_factor": float(branch.get("bayes_factor", np.nan)),
        "max_log_bf_local": float(branch.get("max_log_bf_local", np.nan)),
        "max_event_probability": float(np.nanmax(event_prob)) if event_prob.size and np.isfinite(event_prob).any() else np.nan,
        "production_significant": bool(branch.get("significant", False)),
    }


def _profile_detection_record(base: dict[str, object], profile: dict[str, object], result: dict[str, object]) -> dict[str, object]:
    return {
        **base,
        "trigger_profile": str(profile["trigger_profile"]),
        "trigger_family": str(profile["trigger_family"]),
        "trigger_mode": str(profile["trigger_mode"]),
        "threshold": float(profile["threshold"]),
        "is_production": bool(profile["is_production"]),
        "significant": bool(result["significant"]),
        "event_points": int(len(result["event_indices"])),
        "raw_trigger_points": int(result["raw_trigger_points"]),
        "trigger_max": float(result["trigger_max"]) if np.isfinite(result["trigger_max"]) else np.nan,
        "trigger_threshold": float(result["trigger_threshold"]),
        "run_count": int(result["n_runs"]),
        "max_run_points": int(result["max_run_points"]),
        "max_run_duration": float(result["max_run_duration"]) if np.isfinite(result["max_run_duration"]) else np.nan,
        "max_run_sum": float(result["max_run_sum"]) if np.isfinite(result["max_run_sum"]) else np.nan,
        "max_run_max": float(result["max_run_max"]) if np.isfinite(result["max_run_max"]) else np.nan,
        "max_run_cameras": int(result["max_run_cameras"]),
    }


def _evaluate_task(task: tuple[dict[str, object], dict[str, object]]) -> dict[str, object]:
    row_dict, config_dict = task
    row = pd.Series(row_dict)
    config = March18FlatTriggerConfig(**_dict_to_config_payload(config_dict))
    profiles_by_kind = {str(kind): trigger_profiles(config, kind=str(kind)) for kind in config.event_kinds}
    try:
        raw_df = _read_lightcurve(row["dat_path"])
        valid_mask = (
            np.isfinite(raw_df["JD"])
            & np.isfinite(raw_df["mag"])
            & np.isfinite(raw_df["error"])
            & (raw_df["error"] > 0)
            & (raw_df["error"] < 10)
        )
        df = raw_df.loc[valid_mask].copy()
        pre_bad_cameras: set[object] = set()
        if bool(config_dict["auto_filter_bad_cameras"]) and "camera#" in df.columns:
            df, pre_bad_cameras = filter_bad_cameras(
                df,
                lc_path=str(row["dat_path"]),
                filter_scatter=False,
                filter_offset=False,
                filter_catastrophic=True,
                scatter_ratio_threshold=float(config_dict["bad_camera_scatter_ratio"]),
            )

        baseline_func = _baseline_func_from_name(str(config_dict["baseline_func"]))
        baseline_kwargs = {
            "S0": float(config_dict["baseline_s0"]),
            "w0": float(config_dict["baseline_w0"]),
            "q": float(config_dict["baseline_q"]),
            "jitter": float(config_dict["baseline_jitter"]),
            "sigma_floor": config_dict["baseline_sigma_floor"],
            "add_sigma_eff_col": True,
        }
        scored = score_lightcurve(
            df,
            baseline_func=baseline_func,
            baseline_kwargs=baseline_kwargs,
            filter_residual_bad_cameras_enabled=bool(config_dict["auto_filter_bad_cameras"]),
            bad_camera_scatter_ratio=float(config_dict["bad_camera_scatter_ratio"]),
            p_points=int(config_dict["p_points"]),
            mag_points=int(config_dict["mag_points"]),
            trigger_mode="posterior_prob",
            logbf_threshold_dip=float(config_dict["logbf_threshold_dip"]),
            logbf_threshold_jump=float(config_dict["logbf_threshold_jump"]),
            significance_threshold=float(config_dict["significance_threshold"]),
            run_min_points=int(config_dict["run_min_points"]),
            max_gap_points=int(config_dict["run_max_gap_points"]),
            run_max_gap_days=config_dict["run_max_gap_days"],
            run_min_duration_days=config_dict["run_min_duration_days"],
            compute_event_prob=True,
        )
        clean_df = clean_lc(scored["df"]).reset_index(drop=True)
        jd = clean_df["JD"].to_numpy(dtype=float)
        cam_vec = clean_df["camera#"].to_numpy() if "camera#" in clean_df.columns else None

        score_rows: list[dict[str, object]] = []
        trigger_rows: list[dict[str, object]] = []
        for kind in config_dict["event_kinds"]:
            kind = str(kind)
            branch = scored[kind]
            base = _base_score_record(
                manifest_row=row,
                clean_df=clean_df,
                scored=scored,
                kind=kind,
                pre_bad_cameras=pre_bad_cameras,
            )
            score_rows.append(base)
            log_bf_local = np.asarray(branch.get("log_bf_local", []), dtype=float)
            event_probability = np.asarray(branch.get("event_probability", []), dtype=float)
            for profile in profiles_by_kind[kind]:
                profile_result = _run_profile(
                    profile,
                    jd=jd,
                    cam_vec=cam_vec,
                    log_bf_local=log_bf_local,
                    event_probability=event_probability,
                    config_dict=config_dict,
                    kind=kind,
                )
                trigger_rows.append(_profile_detection_record(base, profile, profile_result))
        return {"scores": score_rows, "triggers": trigger_rows}
    except Exception as exc:
        score_rows = []
        trigger_rows = []
        for kind in config_dict["event_kinds"]:
            base = {
                "source_id": str(row_dict.get("source_id", "")),
                "dat_path": str(row_dict.get("dat_path", "")),
                "mag_bin": str(row_dict.get("mag_bin", "")),
                "event_kind": str(kind),
                "status": "error",
                "error": str(exc),
            }
            score_rows.append(base)
            for profile in profiles_by_kind[str(kind)]:
                trigger_rows.append({**base, **profile})
        return {"scores": score_rows, "triggers": trigger_rows}


def _dict_to_config_payload(config_dict: dict[str, object]) -> dict[str, object]:
    payload = dict(config_dict)
    for key in ("output_base_dir", "flat_lc_dir", "manifest_path", "index_file"):
        if payload.get(key) is not None and not isinstance(payload[key], Path):
            payload[key] = Path(str(payload[key]))
    payload["posterior_probability_thresholds"] = tuple(payload["posterior_probability_thresholds"])
    payload["logbf_thresholds"] = tuple(payload["logbf_thresholds"])
    payload["event_kinds"] = tuple(payload["event_kinds"])
    return payload


def evaluate_manifest(
    manifest: pd.DataFrame,
    config: March18FlatTriggerConfig,
    *,
    score_output_path: Path | None = None,
    trigger_output_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = _config_to_dict(config)
    tasks = [(row.to_dict(), cfg) for _, row in manifest.iterrows()]
    score_rows: list[dict[str, object]] = []
    trigger_rows: list[dict[str, object]] = []

    if config.workers and int(config.workers) > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(config.workers)) as executor:
            futures = [executor.submit(_evaluate_task, task) for task in tasks]
            iterator = as_completed(futures)
            if config.show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="March18 real LC scoring")
            for future in iterator:
                result = future.result()
                score_rows.extend(result["scores"])
                trigger_rows.extend(result["triggers"])
    else:
        iterator = tasks
        if config.show_progress:
            iterator = tqdm(tasks, desc="March18 real LC scoring")
        for task in iterator:
            result = _evaluate_task(task)
            score_rows.extend(result["scores"])
            trigger_rows.extend(result["triggers"])

    score_df = pd.DataFrame(score_rows)
    trigger_df = pd.DataFrame(trigger_rows)
    if score_output_path is not None:
        score_output_path.parent.mkdir(parents=True, exist_ok=True)
        score_df.to_parquet(score_output_path, index=False, compression=PARQUET_OUTPUT_COMPRESSION)
    if trigger_output_path is not None:
        trigger_output_path.parent.mkdir(parents=True, exist_ok=True)
        trigger_df.to_parquet(trigger_output_path, index=False, compression=PARQUET_OUTPUT_COMPRESSION)
    return score_df, trigger_df


def summarize_trigger_results(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["event_kind", "trigger_family", "trigger_profile", "threshold"]
    for key, sub_all in df.groupby(group_cols, dropna=False):
        row = {col: val for col, val in zip(group_cols, key)}
        ok = sub_all[sub_all["status"].eq("ok")].copy()
        sig = ok["significant"].fillna(False).astype(bool) if len(ok) else pd.Series(dtype=bool)
        row.update(
            {
                "n": int(len(sub_all)),
                "n_ok": int(len(ok)),
                "status_error_rate": float(1.0 - len(ok) / max(1, len(sub_all))),
                "significant_count": int(sig.sum()) if len(ok) else 0,
                "significant_rate": float(sig.mean()) if len(ok) else np.nan,
                "median_raw_trigger_points": float(ok["raw_trigger_points"].median()) if len(ok) else np.nan,
                "median_event_points": float(ok["event_points"].median()) if len(ok) else np.nan,
                "median_trigger_max": float(ok["trigger_max"].median()) if len(ok) else np.nan,
                "median_max_event_probability": float(ok["max_event_probability"].median()) if len(ok) else np.nan,
                "median_max_log_bf_local": float(ok["max_log_bf_local"].median()) if len(ok) else np.nan,
                "median_bayes_factor": float(ok["bayes_factor"].median()) if len(ok) else np.nan,
                "median_residual_bad_camera_count": float(ok["residual_bad_camera_count"].median()) if len(ok) else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_pairwise_production_comparison(df: pd.DataFrame) -> pd.DataFrame:
    prod = df[df["is_production"].fillna(False).astype(bool) & df["status"].eq("ok")].copy()
    rows: list[dict[str, object]] = []
    for kind, sub in prod.groupby("event_kind", dropna=False):
        posterior = sub[sub["trigger_family"].eq("loo_posterior_prob")].set_index("source_id")
        logbf = sub[sub["trigger_family"].eq("local_logbf")].set_index("source_id")
        common = posterior.index.intersection(logbf.index)
        if common.empty:
            continue
        pp = posterior.loc[common]
        bf = logbf.loc[common]
        pp_sig = pp["significant"].fillna(False).astype(bool)
        bf_sig = bf["significant"].fillna(False).astype(bool)
        rows.append(
            {
                "event_kind": kind,
                "n_common": int(len(common)),
                "posterior_significant_rate": float(pp_sig.mean()),
                "logbf_significant_rate": float(bf_sig.mean()),
                "both_significant_rate": float((pp_sig & bf_sig).mean()),
                "posterior_only_significant_rate": float((pp_sig & ~bf_sig).mean()),
                "logbf_only_significant_rate": float((~pp_sig & bf_sig).mean()),
                "neither_significant_rate": float((~pp_sig & ~bf_sig).mean()),
                "agreement_rate": float((pp_sig == bf_sig).mean()),
                "median_event_points_delta_posterior_minus_logbf": float((pp["event_points"] - bf["event_points"]).median()),
            }
        )
    return pd.DataFrame(rows)


def run_march18_flat_trigger_mode_benchmark(config: March18FlatTriggerConfig) -> March18FlatTriggerRun:
    run_dir = make_run_dir(config)
    write_config(run_dir, config)
    manifest_path = run_dir / "manifest.parquet"
    score_path = run_dir / "score_results.parquet"
    trigger_path = run_dir / "trigger_results.parquet"

    if manifest_path.exists() and not config.force:
        manifest = pd.read_parquet(manifest_path)
    else:
        manifest = build_manifest(config)
        manifest.to_parquet(manifest_path, index=False, compression=PARQUET_OUTPUT_COMPRESSION)

    if score_path.exists() and trigger_path.exists() and not config.force:
        score_results = pd.read_parquet(score_path)
        trigger_results = pd.read_parquet(trigger_path)
    else:
        score_results, trigger_results = evaluate_manifest(
            manifest,
            config,
            score_output_path=score_path,
            trigger_output_path=trigger_path,
        )

    summary = summarize_trigger_results(trigger_results)
    pairwise = build_pairwise_production_comparison(trigger_results)
    summary_dir = run_dir / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(summary_dir / "all_thresholds_overall.parquet", index=False, compression=PARQUET_OUTPUT_COMPRESSION)
    summary.to_csv(summary_dir / "all_thresholds_overall.csv", index=False)
    pairwise.to_parquet(summary_dir / "pairwise_production.parquet", index=False, compression=PARQUET_OUTPUT_COMPRESSION)
    pairwise.to_csv(summary_dir / "pairwise_production.csv", index=False)

    return March18FlatTriggerRun(
        config=config,
        run_dir=run_dir,
        manifest=manifest,
        score_results=score_results,
        trigger_results=trigger_results,
        summary_overall=summary,
        pairwise_production=pairwise,
    )


def load_march18_flat_trigger_mode_benchmark(run_dir: Path | str) -> March18FlatTriggerRun:
    run_dir = Path(run_dir).expanduser()
    config = load_config(run_dir)
    manifest = pd.read_parquet(run_dir / "manifest.parquet")
    score_results = pd.read_parquet(run_dir / "score_results.parquet")
    trigger_results = pd.read_parquet(run_dir / "trigger_results.parquet")
    summary = summarize_trigger_results(trigger_results)
    pairwise = build_pairwise_production_comparison(trigger_results)
    return March18FlatTriggerRun(
        config=config,
        run_dir=run_dir,
        manifest=manifest,
        score_results=score_results,
        trigger_results=trigger_results,
        summary_overall=summary,
        pairwise_production=pairwise,
    )


def plot_threshold_sweep(summary: pd.DataFrame, *, kind: str = "dip", ax: np.ndarray | None = None) -> np.ndarray:
    metrics = [
        ("significant_rate", "Significant rate"),
        ("median_event_points", "Median event points"),
        ("median_raw_trigger_points", "Median raw trigger points"),
    ]
    if ax is None:
        _, ax = plt.subplots(len(metrics), 2, figsize=(14, 8), sharex="col")
    colors = {"loo_posterior_prob": "goldenrod", "local_logbf": "tab:purple"}
    labels = {"loo_posterior_prob": "LOO posterior probability", "local_logbf": "local log BF"}
    sub_kind = summary[summary["event_kind"].eq(kind)].copy()
    for row_idx, (metric, title) in enumerate(metrics):
        for col_idx, family in enumerate(["loo_posterior_prob", "local_logbf"]):
            family_sub = sub_kind[sub_kind["trigger_family"].eq(family)].sort_values("threshold")
            axis = ax[row_idx, col_idx]
            axis.plot(family_sub["threshold"], family_sub[metric], marker="o", color=colors[family])
            axis.set_title(f"{title}: {labels[family]}")
            axis.grid(alpha=0.25)
            if row_idx == len(metrics) - 1:
                axis.set_xlabel("posterior probability threshold" if family == "loo_posterior_prob" else "local log BF threshold")
    return ax


def plot_score_space(scores: pd.DataFrame, *, kind: str = "dip", ax: plt.Axes | None = None) -> plt.Axes:
    sub = scores[scores["event_kind"].eq(kind) & scores["status"].eq("ok")].copy()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sub["max_log_bf_local"], sub["max_event_probability"], s=10, alpha=0.35)
    ax.axhline(posterior_probability_threshold(SIGNIFICANCE_THRESHOLD), color="goldenrod", lw=1.0, label="posterior production threshold")
    ax.axvline(LOGBF_THRESHOLD_DIP if kind == "dip" else LOGBF_THRESHOLD_JUMP, color="tab:purple", lw=1.0, label="logBF production threshold")
    ax.set_xlabel("max local log BF")
    ax.set_ylabel("max LOO P(event)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Real March 18 score-space relationship: {kind}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return ax
