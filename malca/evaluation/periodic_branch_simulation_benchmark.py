from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import math
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

from malca.baseline import (
    per_camera_gp_baseline_masked,
    phase_template_baseline,
)
from malca.config import (
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
    TRIGGER_MODE,
)
from malca.stv.events import score_lightcurve
from malca.lightcurve_publication import (
    plot_lightcurve_panel,
    plot_phase_panel,
    plot_residual_panel,
)


PERIODIC_BRANCH_MODE_SPECS: dict[str, dict[str, object]] = {
    "phase_template_true_period": {
        "baseline": "phase_template",
        "period_scale": 1.0,
        "period_error_sigma_frac": 0.0,
        "label": "Phase template, true period",
    },
    "phase_template_1pct_period_error": {
        "baseline": "phase_template",
        "period_scale": 1.0,
        "period_error_sigma_frac": 0.01,
        "label": "Phase template, 1% selected-period scatter",
    },
    "phase_template_5pct_period_error": {
        "baseline": "phase_template",
        "period_scale": 1.0,
        "period_error_sigma_frac": 0.05,
        "label": "Phase template, 5% selected-period scatter",
    },
    "phase_template_half_period_alias": {
        "baseline": "phase_template",
        "period_scale": 0.5,
        "period_error_sigma_frac": 0.0,
        "label": "Phase template, half-period alias",
    },
    "phase_template_double_period_alias": {
        "baseline": "phase_template",
        "period_scale": 2.0,
        "period_error_sigma_frac": 0.0,
        "label": "Phase template, double-period alias",
    },
    "gp_masked_control": {
        "baseline": "gp_masked",
        "period_scale": np.nan,
        "period_error_sigma_frac": np.nan,
        "label": "Masked per-camera GP control",
    },
}

DEFAULT_MODE_NAMES: tuple[str, ...] = (
    "phase_template_true_period",
    "phase_template_1pct_period_error",
    "phase_template_5pct_period_error",
    "gp_masked_control",
)


@dataclass
class PeriodicBranchSimConfig:
    output_base_dir: Path = Path("output/diagnostics/periodic_branch_simulation_benchmark")
    run_tag: str | None = None
    n_trials: int = 120000
    seed: int = 20260514
    workers: int = 8
    show_progress: bool = True
    force: bool = False
    mode_names: tuple[str, ...] = DEFAULT_MODE_NAMES

    # Simulation mix. With the defaults, 108000/120000 sources contain an
    # injected one-off dip.
    control_fraction: float = 0.10
    small_dip_fraction: float = 0.50
    medium_dip_fraction: float = 0.32
    broad_dip_fraction: float = 0.08

    # Event-scoring settings. These intentionally default to production values.
    trigger_mode: str = TRIGGER_MODE
    logbf_threshold_dip: float = LOGBF_THRESHOLD_DIP
    logbf_threshold_jump: float = LOGBF_THRESHOLD_JUMP
    significance_threshold: float = SIGNIFICANCE_THRESHOLD
    p_points: int = P_POINTS
    mag_points: int = MAG_POINTS
    run_min_points: int = RUN_MIN_POINTS
    run_max_gap_points: int = RUN_MAX_GAP_POINTS
    run_max_gap_days: float | None = None
    run_min_duration_days: float = 0.0
    compute_event_prob: bool = True

    # Baseline kwargs for parity with events.py defaults.
    baseline_s0: float = BASELINE_S0
    baseline_w0: float = BASELINE_W0
    baseline_q: float = BASELINE_Q
    baseline_jitter: float = BASELINE_JITTER
    baseline_sigma_floor: float | None = None


@dataclass
class PeriodicBranchBenchmarkRun:
    config: PeriodicBranchSimConfig
    run_dir: Path
    trial_design: pd.DataFrame
    branch_results: pd.DataFrame
    summary_overall: pd.DataFrame
    summary_slices: dict[str, pd.DataFrame]


def make_run_dir(config: PeriodicBranchSimConfig) -> Path:
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


def write_config(run_dir: Path, config: PeriodicBranchSimConfig) -> None:
    payload = asdict(config)
    payload["output_base_dir"] = str(payload["output_base_dir"])
    payload["mode_names"] = list(payload["mode_names"])
    with (run_dir / "config.json").open("w", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def _rng_for(seed: int, *tokens: int) -> np.random.Generator:
    value = int(seed) & 0xFFFFFFFF
    for token in tokens:
        value = (value * 1664525 + int(token) + 1013904223) & 0xFFFFFFFF
    return np.random.default_rng(value)


def _stable_text_token(text: str) -> int:
    value = 0
    for char in str(text):
        value = (value * 131 + ord(char)) & 0xFFFFFFFF
    return int(value)


def _log_uniform(rng: np.random.Generator, low: float, high: float, size: int | None = None) -> np.ndarray:
    return 10.0 ** rng.uniform(np.log10(low), np.log10(high), size=size)


def _build_dip_classes(config: PeriodicBranchSimConfig, rng: np.random.Generator) -> np.ndarray:
    classes = np.array(["control_none", "small", "medium", "broad"], dtype=object)
    raw = np.array(
        [
            config.control_fraction,
            config.small_dip_fraction,
            config.medium_dip_fraction,
            config.broad_dip_fraction,
        ],
        dtype=float,
    )
    raw = np.maximum(raw, 0.0)
    if raw.sum() <= 0:
        raw = np.array([0.1, 0.5, 0.32, 0.08], dtype=float)
    probs = raw / raw.sum()
    return rng.choice(classes, size=int(config.n_trials), p=probs)


def generate_trial_design(config: PeriodicBranchSimConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    n = int(config.n_trials)
    dip_classes = _build_dip_classes(config, rng)
    waveform_kinds = np.array(["sinusoid", "double_wave", "spot_like", "broad_eclipse", "mixed"], dtype=object)

    rows: list[dict[str, object]] = []
    for trial_id in range(n):
        trial_rng = _rng_for(config.seed, trial_id)
        dip_class = str(dip_classes[trial_id])
        period_days = float(_log_uniform(trial_rng, 0.8, 80.0))
        n_years = int(trial_rng.integers(2, 8))
        span_days = float(n_years * 365.25 + trial_rng.uniform(-45.0, 90.0))
        n_points_target = int(trial_rng.integers(90, 620))
        n_cameras = int(trial_rng.integers(2, 6))
        n_bands = int(trial_rng.choice([1, 2], p=[0.35, 0.65]))
        base_mag = float(trial_rng.uniform(12.0, 15.0))

        if dip_class == "small":
            dip_amp = float(trial_rng.uniform(0.06, 0.14))
            dip_sigma = float(_log_uniform(trial_rng, 0.25, 3.0))
        elif dip_class == "medium":
            dip_amp = float(trial_rng.uniform(0.14, 0.34))
            dip_sigma = float(_log_uniform(trial_rng, 0.35, 5.5))
        elif dip_class == "broad":
            dip_amp = float(trial_rng.uniform(0.10, 0.28))
            dip_sigma = float(_log_uniform(trial_rng, 4.0, 16.0))
        else:
            dip_amp = 0.0
            dip_sigma = np.nan

        rows.append(
            {
                "trial_id": int(trial_id),
                "trial_seed": int(trial_rng.integers(1, 2**31 - 1)),
                "dip_class": dip_class,
                "has_dip": bool(dip_class != "control_none"),
                "period_days": period_days,
                "span_days": span_days,
                "n_years": n_years,
                "n_points_target": n_points_target,
                "n_cameras": n_cameras,
                "n_bands": n_bands,
                "base_mag": base_mag,
                "waveform_kind": str(trial_rng.choice(waveform_kinds)),
                "periodic_amp_mag": float(trial_rng.uniform(0.04, 0.36)),
                "secondary_harmonic_frac": float(trial_rng.uniform(0.05, 0.45)),
                "phase_wander_cycles": float(trial_rng.uniform(0.0, 0.055)),
                "amp_mod_frac": float(trial_rng.uniform(0.0, 0.35)),
                "slow_drift_amp_mag": float(trial_rng.uniform(0.0, 0.10)),
                "quasi_noise_amp_mag": float(trial_rng.uniform(0.0, 0.07)),
                "camera_offset_sigma_mag": float(trial_rng.uniform(0.015, 0.13)),
                "season_zero_point_sigma_mag": float(trial_rng.uniform(0.0, 0.055)),
                "visible_fraction": float(trial_rng.uniform(0.34, 0.63)),
                "weather_dropout_fraction": float(trial_rng.uniform(0.05, 0.32)),
                "error_floor_mag": float(trial_rng.uniform(0.012, 0.040) * (1.0 + max(base_mag - 13.0, 0.0) * 0.28)),
                "extra_white_noise_mag": float(trial_rng.uniform(0.004, 0.045)),
                "outlier_fraction": float(trial_rng.choice([0.0, trial_rng.uniform(0.002, 0.018)], p=[0.65, 0.35])),
                "dip_amp_mag": dip_amp,
                "dip_sigma_days": dip_sigma,
                "dip_asymmetry": float(trial_rng.uniform(0.45, 2.6)),
                "dip_shape_power": float(trial_rng.uniform(1.6, 3.4)),
            }
        )

    design = pd.DataFrame(rows)
    design["dip_amp_bin"] = pd.cut(
        design["dip_amp_mag"],
        bins=[-np.inf, 0.0, 0.10, 0.18, 0.26, np.inf],
        labels=["control", "0.06-0.10", "0.10-0.18", "0.18-0.26", "0.26+"],
    ).astype(str)
    design["period_bin"] = pd.cut(
        design["period_days"],
        bins=[0.0, 2.0, 5.0, 10.0, 25.0, 50.0, np.inf],
        labels=["<2d", "2-5d", "5-10d", "10-25d", "25-50d", "50d+"],
    ).astype(str)
    design["points_bin"] = pd.cut(
        design["n_points_target"],
        bins=[0, 120, 220, 360, 520, np.inf],
        labels=["<120", "120-220", "220-360", "360-520", "520+"],
    ).astype(str)
    return design


def _allocate_counts(rng: np.random.Generator, total: int, n_groups: int) -> np.ndarray:
    weights = rng.gamma(shape=1.6, scale=1.0, size=int(n_groups))
    weights = weights / weights.sum()
    return rng.multinomial(int(total), weights)


def _sample_camera_times(
    rng: np.random.Generator,
    *,
    n_points: int,
    span_days: float,
    visible_fraction: float,
    weather_dropout_fraction: float,
    season_anchor: float,
) -> np.ndarray:
    n_years = max(1, int(math.ceil(span_days / 365.25)))
    visible_days = float(np.clip(visible_fraction, 0.15, 0.85) * 365.25)
    active = rng.random(n_years) > 0.10
    if not active.any():
        active[int(rng.integers(0, n_years))] = True
    weights = active.astype(float) * rng.gamma(1.5, 1.0, n_years)
    if weights.sum() <= 0:
        weights[:] = 1.0
    counts = rng.multinomial(int(n_points), weights / weights.sum())

    times: list[np.ndarray] = []
    for year_idx, count in enumerate(counts):
        if count <= 0:
            continue
        start = year_idx * 365.25 + season_anchor + rng.normal(0.0, 13.0)
        start = max(0.0, min(start, span_days))
        stop = min(span_days, start + visible_days * rng.uniform(0.75, 1.15))
        if stop <= start:
            continue
        n_nights = max(1, int(np.ceil(count / rng.uniform(1.0, 2.7))))
        nights = rng.uniform(start, stop, n_nights)
        keep = rng.random(n_nights) > weather_dropout_fraction
        if keep.sum() == 0:
            keep[int(rng.integers(0, n_nights))] = True
        nights = nights[keep]
        visits_per_night = rng.choice([1, 1, 1, 2, 2, 3], size=nights.size)
        sampled = np.repeat(nights, visits_per_night)
        sampled = sampled + rng.normal(0.0, 0.055, sampled.size)
        if sampled.size > count:
            sampled = rng.choice(sampled, size=count, replace=False)
        times.append(sampled)

    if times:
        out = np.concatenate(times)
    else:
        out = np.empty(0, dtype=float)

    attempts = 0
    while out.size < int(n_points) and attempts < 20:
        attempts += 1
        year_idx = int(rng.integers(0, n_years))
        start = year_idx * 365.25 + season_anchor
        stop = min(span_days, start + visible_days)
        if stop > start:
            extra = rng.uniform(start, stop, int(n_points) - out.size)
            out = np.concatenate([out, extra])

    out = out[np.isfinite(out)]
    out = out[(out >= 0.0) & (out <= span_days)]
    if out.size > int(n_points):
        out = rng.choice(out, size=int(n_points), replace=False)
    return np.sort(out)


def _phase_distance(phase: np.ndarray, center: float) -> np.ndarray:
    return ((phase - center + 0.5) % 1.0) - 0.5


def _periodic_waveform(phase: np.ndarray, row: pd.Series, rng: np.random.Generator) -> np.ndarray:
    kind = str(row["waveform_kind"])
    amp = float(row["periodic_amp_mag"])
    h2 = float(row["secondary_harmonic_frac"])
    phi2 = float(rng.uniform(0.0, 2.0 * np.pi))
    phi3 = float(rng.uniform(0.0, 2.0 * np.pi))

    if kind == "sinusoid":
        raw = np.sin(2.0 * np.pi * phase + phi2) + 0.18 * np.sin(4.0 * np.pi * phase + phi3)
    elif kind == "double_wave":
        raw = 0.75 * np.sin(2.0 * np.pi * phase + phi2) + h2 * np.sin(4.0 * np.pi * phase + phi3)
    elif kind == "spot_like":
        raw = np.sin(2.0 * np.pi * phase + phi2) + 0.32 * np.sin(4.0 * np.pi * phase + phi3)
        raw += 0.35 * np.maximum(0.0, np.sin(2.0 * np.pi * phase + phi2)) ** 2
    elif kind == "broad_eclipse":
        primary = np.exp(-0.5 * (_phase_distance(phase, rng.uniform(0.15, 0.35)) / rng.uniform(0.055, 0.12)) ** 2)
        secondary = 0.35 * np.exp(-0.5 * (_phase_distance(phase, rng.uniform(0.62, 0.82)) / rng.uniform(0.06, 0.16)) ** 2)
        raw = primary + secondary + 0.12 * np.sin(2.0 * np.pi * phase + phi2)
    else:
        primary = 0.75 * np.sin(2.0 * np.pi * phase + phi2)
        secondary = h2 * np.sin(4.0 * np.pi * phase + phi3)
        bump = 0.45 * np.exp(-0.5 * (_phase_distance(phase, rng.uniform(0.1, 0.9)) / rng.uniform(0.04, 0.11)) ** 2)
        raw = primary + secondary + bump

    raw = raw - float(np.nanmedian(raw))
    scale = np.nanpercentile(np.abs(raw), 95)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return amp * raw / scale


def simulate_periodic_lightcurve(row: pd.Series | dict[str, object]) -> pd.DataFrame:
    row = pd.Series(row)
    rng = np.random.default_rng(int(row["trial_seed"]))
    n_cameras = int(row["n_cameras"])
    n_target = int(row["n_points_target"])
    counts = _allocate_counts(rng, n_target, n_cameras)
    season_anchor = float(rng.uniform(0.0, 180.0))

    records: list[pd.DataFrame] = []
    camera_offsets = rng.normal(0.0, float(row["camera_offset_sigma_mag"]), n_cameras)
    camera_noise_scale = rng.uniform(0.8, 1.35, n_cameras)
    camera_active_shift = rng.normal(0.0, 9.0, n_cameras)
    band_offset_value = float(rng.normal(0.34, 0.08)) if int(row["n_bands"]) == 2 else 0.0

    for camera_idx, count in enumerate(counts):
        if count <= 0:
            continue
        times_rel = _sample_camera_times(
            rng,
            n_points=int(count),
            span_days=float(row["span_days"]),
            visible_fraction=float(row["visible_fraction"]),
            weather_dropout_fraction=float(row["weather_dropout_fraction"]),
            season_anchor=season_anchor + camera_active_shift[camera_idx],
        )
        if times_rel.size == 0:
            continue
        band = rng.choice([0, 1], size=times_rel.size, p=[0.62, 0.38]) if int(row["n_bands"]) == 2 else np.zeros(times_rel.size, dtype=int)
        records.append(
            pd.DataFrame(
                {
                    "time_rel_days": times_rel,
                    "camera#": np.full(times_rel.size, camera_idx + 1, dtype=int),
                    "camera_name": np.full(times_rel.size, f"cam{camera_idx + 1}", dtype=object),
                    "v_g_band": band.astype(int),
                    "_camera_offset": np.full(times_rel.size, camera_offsets[camera_idx], dtype=float),
                    "_camera_noise_scale": np.full(times_rel.size, camera_noise_scale[camera_idx], dtype=float),
                }
            )
        )

    if not records:
        raise ValueError(f"trial {row.get('trial_id', 'unknown')} produced no observation times")

    df = pd.concat(records, ignore_index=True)
    df = df.sort_values("time_rel_days").reset_index(drop=True)
    n = len(df)
    jd0 = 2458000.0 + rng.uniform(0.0, 220.0)
    t_rel = df["time_rel_days"].to_numpy(dtype=float)
    jd = jd0 + t_rel
    span = max(float(row["span_days"]), 1.0)
    period = float(row["period_days"])

    phase_wander = float(row["phase_wander_cycles"]) * np.sin(
        2.0 * np.pi * t_rel / max(span * rng.uniform(0.7, 1.8), period) + rng.uniform(0.0, 2.0 * np.pi)
    )
    phase = np.mod(t_rel / period + phase_wander, 1.0)
    waveform = _periodic_waveform(phase, row, rng)
    amp_mod = 1.0 + float(row["amp_mod_frac"]) * np.sin(
        2.0 * np.pi * t_rel / max(span * rng.uniform(0.55, 1.4), period) + rng.uniform(0.0, 2.0 * np.pi)
    )
    slow_drift = float(row["slow_drift_amp_mag"]) * np.sin(
        2.0 * np.pi * t_rel / max(span * rng.uniform(0.8, 2.0), period) + rng.uniform(0.0, 2.0 * np.pi)
    )
    slow_drift += rng.normal(0.0, 0.025) * (t_rel - np.nanmedian(t_rel)) / span

    n_knots = max(5, int(span // 120) + 3)
    knots = np.linspace(0.0, span, n_knots)
    knot_values = rng.normal(0.0, float(row["quasi_noise_amp_mag"]), n_knots)
    quasi = np.interp(t_rel, knots, knot_values)

    year_index = np.floor(t_rel / 365.25).astype(int)
    season_key = pd.Series(list(zip(df["camera#"].astype(int), year_index)))
    unique_keys = season_key.drop_duplicates().tolist()
    season_offsets = {key: rng.normal(0.0, float(row["season_zero_point_sigma_mag"])) for key in unique_keys}
    seasonal = np.array([season_offsets[key] for key in season_key], dtype=float)

    band_offset = np.where(df["v_g_band"].to_numpy(dtype=int) == 1, band_offset_value, 0.0)
    true_baseline = (
        float(row["base_mag"])
        + amp_mod * waveform
        + slow_drift
        + quasi
        + df["_camera_offset"].to_numpy(dtype=float)
        + band_offset
        + seasonal
    )

    dip_signal = np.zeros(n, dtype=float)
    dip_t0 = np.nan
    dip_window_days = np.nan
    has_dip = bool(row["has_dip"])
    if has_dip:
        central = df[(df["time_rel_days"] > span * 0.08) & (df["time_rel_days"] < span * 0.92)]
        source = central if len(central) else df
        t0_rel = float(source["time_rel_days"].sample(n=1, random_state=int(rng.integers(1, 2**31 - 1))).iloc[0])
        t0_rel += float(rng.normal(0.0, max(float(row["dip_sigma_days"]), 0.2) * 0.35))
        dip_t0 = jd0 + t0_rel
        dt = jd - dip_t0
        sigma = max(float(row["dip_sigma_days"]), 0.05)
        asymmetry = max(float(row["dip_asymmetry"]), 0.05)
        sigma_left = sigma / math.sqrt(asymmetry)
        sigma_right = sigma * math.sqrt(asymmetry)
        width = np.where(dt < 0.0, sigma_left, sigma_right)
        power = float(row["dip_shape_power"])
        dip_signal = float(row["dip_amp_mag"]) * np.exp(-0.5 * np.abs(dt / width) ** power)
        dip_window_days = float(2.8 * max(sigma_left, sigma_right))

    err = float(row["error_floor_mag"]) * df["_camera_noise_scale"].to_numpy(dtype=float)
    err = err * rng.lognormal(mean=0.0, sigma=0.25, size=n)
    err = np.maximum(err, 0.004)
    white_noise = rng.normal(0.0, np.sqrt(err**2 + float(row["extra_white_noise_mag"]) ** 2), n)
    mag = true_baseline + dip_signal + white_noise

    outlier_fraction = float(row["outlier_fraction"])
    if outlier_fraction > 0:
        outlier_mask = rng.random(n) < outlier_fraction
        outlier_delta = rng.normal(0.0, rng.uniform(0.12, 0.45), n)
        mag = mag + outlier_mask.astype(float) * outlier_delta
        err = np.where(outlier_mask, err * rng.uniform(1.2, 2.8), err)

    truth_threshold = max(0.025, 0.25 * float(row["dip_amp_mag"]))
    truth_dip_mask = dip_signal >= truth_threshold
    truth_peak_snr = float(np.nanmax(dip_signal / err)) if has_dip and n else 0.0
    truth_support_points = int(np.count_nonzero(truth_dip_mask))
    truth_observable = bool(has_dip and truth_support_points >= RUN_MIN_POINTS and truth_peak_snr >= 2.5)

    df["JD"] = jd
    df["mag"] = mag
    df["error"] = err
    df["field"] = "sim_field"
    df["saturated"] = 0
    df["point_id"] = np.arange(n, dtype=int)
    df["true_baseline_mag"] = true_baseline
    df["truth_dip_signal"] = dip_signal
    df["truth_dip_mask"] = truth_dip_mask
    df["truth_dip_t0"] = dip_t0
    df["truth_dip_window_days"] = dip_window_days
    df["truth_peak_snr"] = truth_peak_snr
    df["truth_support_points"] = truth_support_points
    df["truth_observable"] = truth_observable
    df["trial_id"] = int(row["trial_id"])
    df["period_days"] = period
    df["dip_class"] = str(row["dip_class"])
    return df.drop(columns=["_camera_offset", "_camera_noise_scale"])


def _mode_selected_period(row: pd.Series, mode_name: str) -> float:
    spec = PERIODIC_BRANCH_MODE_SPECS[mode_name]
    if spec["baseline"] != "phase_template":
        return np.nan
    scale = float(spec.get("period_scale", 1.0))
    sigma = float(spec.get("period_error_sigma_frac", 0.0))
    period = float(row["period_days"]) * scale
    if sigma > 0:
        rng = _rng_for(int(row["trial_seed"]), _stable_text_token(mode_name))
        period *= 1.0 + float(rng.normal(0.0, sigma))
    return float(max(period, 1e-6))


def _baseline_for_mode(mode_name: str) -> tuple[object, dict[str, object], float]:
    spec = PERIODIC_BRANCH_MODE_SPECS[mode_name]
    if spec["baseline"] == "phase_template":
        return phase_template_baseline, {}, np.nan
    if spec["baseline"] == "gp_masked":
        return per_camera_gp_baseline_masked, {}, np.nan
    raise ValueError(f"Unsupported mode: {mode_name}")


def _evaluate_task(task: tuple[dict[str, object], str, dict[str, object]]) -> dict[str, object]:
    row_dict, mode_name, config_dict = task
    row = pd.Series(row_dict)
    try:
        df = simulate_periodic_lightcurve(row)
        selected_period = _mode_selected_period(row, mode_name)
        baseline_func, baseline_kwargs, _ = _baseline_for_mode(mode_name)
        baseline_kwargs = dict(
            S0=float(config_dict["baseline_s0"]),
            w0=float(config_dict["baseline_w0"]),
            q=float(config_dict["baseline_q"]),
            jitter=float(config_dict["baseline_jitter"]),
            sigma_floor=config_dict["baseline_sigma_floor"],
            add_sigma_eff_col=True,
        )
        if PERIODIC_BRANCH_MODE_SPECS[mode_name]["baseline"] == "phase_template":
            baseline_kwargs["period_days"] = selected_period

        scored = score_lightcurve(
            df,
            baseline_func=baseline_func,
            baseline_kwargs=baseline_kwargs,
            p_points=int(config_dict["p_points"]),
            mag_points=int(config_dict["mag_points"]),
            trigger_mode=str(config_dict["trigger_mode"]),
            logbf_threshold_dip=float(config_dict["logbf_threshold_dip"]),
            logbf_threshold_jump=float(config_dict["logbf_threshold_jump"]),
            significance_threshold=float(config_dict["significance_threshold"]),
            run_min_points=int(config_dict["run_min_points"]),
            max_gap_points=int(config_dict["run_max_gap_points"]),
            run_max_gap_days=config_dict["run_max_gap_days"],
            run_min_duration_days=config_dict["run_min_duration_days"],
            compute_event_prob=bool(config_dict["compute_event_prob"]),
        )
        clean_df = scored["df"].reset_index(drop=True)
        df_base = scored["df_base"].reset_index(drop=True)
        dip = scored["dip"]

        event_indices = np.asarray(dip.get("event_indices", []), dtype=int)
        event_indices = event_indices[(event_indices >= 0) & (event_indices < len(clean_df))]
        event_mask = np.zeros(len(clean_df), dtype=bool)
        event_mask[event_indices] = True
        truth_mask = clean_df["truth_dip_mask"].to_numpy(dtype=bool)
        has_dip = bool(row["has_dip"])
        truth_observable = bool(clean_df["truth_observable"].iloc[0]) if len(clean_df) else False
        overlap_points = int(np.count_nonzero(event_mask & truth_mask))
        detected_overlap = bool(overlap_points > 0)
        dip_significant = bool(dip.get("significant", False))
        target_recovered = bool(has_dip and dip_significant and detected_overlap)
        false_positive = bool((not has_dip) and dip_significant)
        off_target_detection = bool(has_dip and dip_significant and not detected_overlap)

        outside = ~truth_mask
        if not outside.any():
            outside = np.ones(len(clean_df), dtype=bool)
        baseline_values = pd.to_numeric(df_base["baseline"], errors="coerce").to_numpy(dtype=float)
        true_baseline = pd.to_numeric(clean_df["true_baseline_mag"], errors="coerce").to_numpy(dtype=float)
        baseline_error = baseline_values - true_baseline
        resid = pd.to_numeric(df_base["resid"], errors="coerce").to_numpy(dtype=float)
        truth_signal = pd.to_numeric(clean_df["truth_dip_signal"], errors="coerce").to_numpy(dtype=float)
        recovered_amp = float(np.nanmax(resid[truth_mask])) if truth_mask.any() and np.isfinite(resid[truth_mask]).any() else np.nan
        true_amp_sampled = float(np.nanmax(truth_signal)) if np.isfinite(truth_signal).any() else np.nan

        baseline_source = str(dip.get("baseline_source", "unknown"))
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
            "truth_peak_snr_actual": float(np.nanmax(truth_signal / clean_df["error"].to_numpy(dtype=float))) if has_dip and len(clean_df) else 0.0,
            "truth_observable_actual": truth_observable,
            "dip_significant": dip_significant,
            "target_recovered": target_recovered,
            "false_positive": false_positive,
            "off_target_detection": off_target_detection,
            "detected_overlap": detected_overlap,
            "overlap_points": overlap_points,
            "event_points": int(event_mask.sum()),
            "dip_run_count": int(dip.get("n_runs", 0)),
            "dip_count": int(len(event_indices)),
            "dip_max_log_bf_local": float(dip.get("max_log_bf_local", np.nan)),
            "dip_bayes_factor": float(dip.get("bayes_factor", np.nan)),
            "dip_trigger_max": float(dip.get("trigger_max", np.nan)),
            "dip_max_run_points": int(dip.get("max_run_points", 0)),
            "dip_max_run_duration": float(dip.get("max_run_duration", np.nan)),
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
    except Exception as exc:
        return {
            "trial_id": int(row_dict.get("trial_id", -1)),
            "mode": mode_name,
            "mode_label": str(PERIODIC_BRANCH_MODE_SPECS.get(mode_name, {}).get("label", mode_name)),
            "status": "error",
            "error": str(exc),
        }


def _config_to_worker_dict(config: PeriodicBranchSimConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["output_base_dir"] = str(payload["output_base_dir"])
    payload["mode_names"] = list(payload["mode_names"])
    return payload


def evaluate_design(
    design: pd.DataFrame,
    config: PeriodicBranchSimConfig,
    *,
    output_path: Path | None = None,
) -> pd.DataFrame:
    tasks: list[tuple[dict[str, object], str, dict[str, object]]] = []
    cfg = _config_to_worker_dict(config)
    for _, row in design.iterrows():
        row_dict = row.to_dict()
        for mode_name in config.mode_names:
            if mode_name not in PERIODIC_BRANCH_MODE_SPECS:
                raise ValueError(f"Unknown benchmark mode: {mode_name}")
            tasks.append((row_dict, str(mode_name), cfg))

    results: list[dict[str, object]] = []
    if config.workers and int(config.workers) > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=int(config.workers)) as executor:
            futures = [executor.submit(_evaluate_task, task) for task in tasks]
            iterator = as_completed(futures)
            if config.show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="Periodic branch simulations")
            for future in iterator:
                results.append(future.result())
    else:
        iterator = tasks
        if config.show_progress:
            iterator = tqdm(tasks, desc="Periodic branch simulations")
        for task in iterator:
            results.append(_evaluate_task(task))

    out = pd.DataFrame(results)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(output_path, index=False, compression=PARQUET_OUTPUT_COMPRESSION)
    return out


def add_metric_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["period_bin"] = pd.cut(
        out["period_days"],
        bins=[0.0, 2.0, 5.0, 10.0, 25.0, 50.0, np.inf],
        labels=["<2d", "2-5d", "5-10d", "10-25d", "25-50d", "50d+"],
    ).astype(str)
    out["dip_amp_bin"] = pd.cut(
        out["dip_amp_mag"],
        bins=[-np.inf, 0.0, 0.10, 0.18, 0.26, np.inf],
        labels=["control", "0.06-0.10", "0.10-0.18", "0.18-0.26", "0.26+"],
    ).astype(str)
    out["width_bin"] = pd.cut(
        out["dip_sigma_days"],
        bins=[-np.inf, 0.5, 1.5, 4.0, 10.0, np.inf],
        labels=["<0.5d", "0.5-1.5d", "1.5-4d", "4-10d", "10d+"],
    ).astype(str)
    out["points_bin_actual"] = pd.cut(
        out["n_points_actual"],
        bins=[0, 120, 220, 360, 520, np.inf],
        labels=["<120", "120-220", "220-360", "360-520", "520+"],
    ).astype(str)
    out["period_error_bin"] = pd.cut(
        out["period_frac_error"],
        bins=[-np.inf, 0.001, 0.01, 0.03, 0.08, np.inf],
        labels=["none/<0.1%", "0.1-1%", "1-3%", "3-8%", "8%+"],
    ).astype(str)
    return out


def merge_design_results(design: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    merged = results.merge(design, on="trial_id", how="left", suffixes=("", "_design"))
    return add_metric_bins(merged)


def _rate(series: pd.Series) -> float:
    if len(series) == 0:
        return np.nan
    return float(series.fillna(False).astype(bool).mean())


def summarize_results(df: pd.DataFrame, group_cols: list[str] | tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ok = df[df["status"].eq("ok")].copy()
    grouped = ok.groupby(list(group_cols), dropna=False) if group_cols else [((), ok)]
    for key, sub in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {col: val for col, val in zip(group_cols, key_tuple)}
        has_dip = sub["has_dip"].fillna(False).astype(bool)
        observable = has_dip & sub["truth_observable_actual"].fillna(False).astype(bool)
        controls = ~has_dip
        detected = sub["dip_significant"].fillna(False).astype(bool)
        recovered = sub["target_recovered"].fillna(False).astype(bool)
        true_positive_detected = detected & has_dip & sub["detected_overlap"].fillna(False).astype(bool)

        row.update(
            {
                "n": int(len(sub)),
                "n_dip": int(has_dip.sum()),
                "n_observable_dip": int(observable.sum()),
                "n_control": int(controls.sum()),
                "status_error_rate": float(1.0 - len(sub) / max(1, len(df.loc[sub.index]))),
                "dip_significant_rate": _rate(detected),
                "observable_recall": float(recovered[observable].mean()) if observable.any() else np.nan,
                "all_dip_recall": float(recovered[has_dip].mean()) if has_dip.any() else np.nan,
                "control_false_positive_rate": float(detected[controls].mean()) if controls.any() else np.nan,
                "off_target_detection_rate": _rate(sub.loc[has_dip, "off_target_detection"]) if has_dip.any() else np.nan,
                "precision_by_trial": float(true_positive_detected.sum() / detected.sum()) if detected.any() else np.nan,
                "phase_template_fallback_rate": _rate(sub["phase_template_fallback"]),
                "median_baseline_mae_outside_dip": float(sub["baseline_mae_outside_dip"].median()),
                "median_baseline_rmse_outside_dip": float(sub["baseline_rmse_outside_dip"].median()),
                "median_resid_rms_outside_dip": float(sub["resid_rms_outside_dip"].median()),
                "median_amp_recovery_ratio": float(sub.loc[has_dip, "amp_recovery_ratio"].median()) if has_dip.any() else np.nan,
                "median_dip_bayes_factor": float(sub["dip_bayes_factor"].median()),
                "median_dip_max_log_bf_local": float(sub["dip_max_log_bf_local"].median()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_summary_slices(merged: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "overall_by_mode": summarize_results(merged, ["mode", "mode_label"]),
        "by_mode_and_dip_class": summarize_results(merged, ["mode", "dip_class"]),
        "by_mode_and_amp": summarize_results(merged, ["mode", "dip_amp_bin"]),
        "by_mode_and_period": summarize_results(merged, ["mode", "period_bin"]),
        "by_mode_and_width": summarize_results(merged, ["mode", "width_bin"]),
        "by_mode_and_points": summarize_results(merged, ["mode", "points_bin_actual"]),
        "by_mode_and_waveform": summarize_results(merged, ["mode", "waveform_kind"]),
        "by_mode_and_period_error": summarize_results(merged, ["mode", "period_error_bin"]),
    }


def run_periodic_branch_simulation_benchmark(config: PeriodicBranchSimConfig) -> PeriodicBranchBenchmarkRun:
    run_dir = make_run_dir(config)
    write_config(run_dir, config)

    design_path = run_dir / "trial_design.parquet"
    results_path = run_dir / "branch_results.parquet"

    if design_path.exists() and not config.force:
        design = pd.read_parquet(design_path)
    else:
        design = generate_trial_design(config)
        design.to_parquet(design_path, index=False, compression=PARQUET_OUTPUT_COMPRESSION)

    if results_path.exists() and not config.force:
        results = pd.read_parquet(results_path)
    else:
        results = evaluate_design(design, config, output_path=results_path)

    merged = merge_design_results(design, results)
    merged_path = run_dir / "branch_results_with_design.parquet"
    merged.to_parquet(merged_path, index=False, compression=PARQUET_OUTPUT_COMPRESSION)

    summary_slices = build_summary_slices(merged)
    summary_dir = run_dir / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for name, table in summary_slices.items():
        table.to_parquet(summary_dir / f"{name}.parquet", index=False, compression=PARQUET_OUTPUT_COMPRESSION)
        table.to_csv(summary_dir / f"{name}.csv", index=False)

    return PeriodicBranchBenchmarkRun(
        config=config,
        run_dir=run_dir,
        trial_design=design,
        branch_results=merged,
        summary_overall=summary_slices["overall_by_mode"],
        summary_slices=summary_slices,
    )


def load_periodic_branch_simulation_benchmark(run_dir: Path | str) -> PeriodicBranchBenchmarkRun:
    run_dir = Path(run_dir).expanduser()
    with (run_dir / "config.json").open("r", encoding="ascii") as handle:
        cfg_payload = json.load(handle)
    cfg_payload["output_base_dir"] = Path(cfg_payload["output_base_dir"])
    cfg_payload["mode_names"] = tuple(cfg_payload["mode_names"])
    config = PeriodicBranchSimConfig(**cfg_payload)
    design = pd.read_parquet(run_dir / "trial_design.parquet")
    merged = pd.read_parquet(run_dir / "branch_results_with_design.parquet")
    summary_slices = build_summary_slices(merged)
    return PeriodicBranchBenchmarkRun(
        config=config,
        run_dir=run_dir,
        trial_design=design,
        branch_results=merged,
        summary_overall=summary_slices["overall_by_mode"],
        summary_slices=summary_slices,
    )


def metric_heatmap(
    df: pd.DataFrame,
    *,
    mode: str,
    row: str,
    col: str,
    metric: str = "target_recovered",
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    sub = df[(df["mode"] == mode) & df["status"].eq("ok")].copy()
    if metric in {"target_recovered", "dip_significant", "false_positive", "off_target_detection"}:
        values = (
            sub.groupby([row, col], dropna=False)[metric]
            .mean()
            .unstack(col)
        )
    else:
        values = (
            sub.groupby([row, col], dropna=False)[metric]
            .median()
            .unstack(col)
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(values.to_numpy(dtype=float), aspect="auto", origin="lower", cmap="viridis", vmin=0 if metric in {"target_recovered", "dip_significant", "false_positive", "off_target_detection"} else None)
    ax.set_xticks(np.arange(values.shape[1]), labels=[str(x) for x in values.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(values.shape[0]), labels=[str(x) for x in values.index])
    ax.set_xlabel(col)
    ax.set_ylabel(row)
    ax.set_title(title or f"{metric} for {mode}")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label(metric)
    return ax


def plot_overall_metrics(summary: pd.DataFrame, *, ax: plt.Axes | None = None) -> plt.Axes:
    metrics = [
        "observable_recall",
        "precision_by_trial",
        "control_false_positive_rate",
        "off_target_detection_rate",
        "phase_template_fallback_rate",
    ]
    plot_df = summary.set_index("mode")[metrics]
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("rate")
    ax.set_title("Periodic-branch recovery and contamination metrics")
    ax.legend(loc="best", fontsize=8)
    ax.tick_params(axis="x", rotation=35)
    return ax


def plot_metric_distributions(
    df: pd.DataFrame,
    *,
    metric: str,
    modes: list[str] | tuple[str, ...] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    ok = df[df["status"].eq("ok")].copy()
    if modes is not None:
        ok = ok[ok["mode"].isin(modes)]
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    for mode, sub in ok.groupby("mode", sort=False):
        vals = pd.to_numeric(sub[metric], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        ax.hist(vals, bins=40, alpha=0.35, density=True, label=str(mode))
    ax.set_xlabel(metric)
    ax.set_ylabel("density")
    ax.set_title(f"Distribution of {metric}")
    ax.legend(fontsize=8)
    return ax


def recompute_trial_for_plot(
    run: PeriodicBranchBenchmarkRun,
    trial_id: int,
    mode: str = "phase_template_true_period",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    row = run.trial_design.loc[run.trial_design["trial_id"] == int(trial_id)]
    if row.empty:
        raise KeyError(f"trial_id {trial_id} not found")
    row_series = row.iloc[0]
    df = simulate_periodic_lightcurve(row_series)
    selected_period = _mode_selected_period(row_series, mode)
    spec = PERIODIC_BRANCH_MODE_SPECS[mode]
    if spec["baseline"] == "phase_template":
        baseline_func = phase_template_baseline
        baseline_kwargs = {
            "period_days": selected_period,
            "S0": run.config.baseline_s0,
            "w0": run.config.baseline_w0,
            "q": run.config.baseline_q,
            "jitter": run.config.baseline_jitter,
            "sigma_floor": run.config.baseline_sigma_floor,
            "add_sigma_eff_col": True,
        }
    else:
        baseline_func = per_camera_gp_baseline_masked
        baseline_kwargs = {
            "S0": run.config.baseline_s0,
            "w0": run.config.baseline_w0,
            "q": run.config.baseline_q,
            "jitter": run.config.baseline_jitter,
            "sigma_floor": run.config.baseline_sigma_floor,
            "add_sigma_eff_col": True,
        }
    scored = score_lightcurve(
        df,
        baseline_func=baseline_func,
        baseline_kwargs=baseline_kwargs,
        p_points=run.config.p_points,
        mag_points=run.config.mag_points,
        trigger_mode=run.config.trigger_mode,
        logbf_threshold_dip=run.config.logbf_threshold_dip,
        logbf_threshold_jump=run.config.logbf_threshold_jump,
        significance_threshold=run.config.significance_threshold,
        run_min_points=run.config.run_min_points,
        max_gap_points=run.config.run_max_gap_points,
        run_max_gap_days=run.config.run_max_gap_days,
        run_min_duration_days=run.config.run_min_duration_days,
        compute_event_prob=run.config.compute_event_prob,
    )
    clean_df = scored["df"].reset_index(drop=True)
    df_base = scored["df_base"].reset_index(drop=True)
    info = {
        "selected_period_days": selected_period,
        "dip": scored["dip"],
        "jump": scored["jump"],
        "mode": mode,
        "mode_label": PERIODIC_BRANCH_MODE_SPECS[mode]["label"],
    }
    return clean_df, df_base, info


def plot_trial_diagnostic(
    run: PeriodicBranchBenchmarkRun,
    trial_id: int,
    *,
    mode: str = "phase_template_true_period",
    ax: np.ndarray | None = None,
) -> np.ndarray:
    df, df_base, info = recompute_trial_for_plot(run, trial_id, mode=mode)
    if ax is None:
        _, ax = plt.subplots(3, 1, figsize=(13, 10), sharex=False)
    jd = df["JD"].to_numpy(dtype=float)
    truth_mask = df["truth_dip_mask"].to_numpy(dtype=bool)
    event_idx = np.asarray(info["dip"].get("event_indices", []), dtype=int)
    event_idx = event_idx[(event_idx >= 0) & (event_idx < len(df))]
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
        ax[0].scatter(jd[truth_mask], df.loc[truth_mask, "mag"], s=28, facecolors="none", edgecolors="limegreen", label="truth dip support")
    if event_idx.size:
        ax[0].scatter(jd[event_idx], df.loc[event_idx, "mag"], marker="x", s=42, color="gold", label="detected event points")
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
        ax[1].scatter(jd[truth_mask], df_base.loc[truth_mask, "resid"], s=28, facecolors="none", edgecolors="limegreen")
    if event_idx.size:
        ax[1].scatter(jd[event_idx], df_base.loc[event_idx, "resid"], marker="x", s=42, color="gold")
    ax[1].set_title("Residuals used by Bayesian event scoring")

    plot_phase_panel(
        ax[2],
        df,
        period_days=float(df["period_days"].iloc[0]),
        epoch_jd=float(np.nanmin(jd)),
        group_by="none",
        show_errorbars=False,
        marker_size=3.4,
        legend="none",
    )
    order = np.argsort(phase)
    ax[2].plot(phase[order], df_base["baseline"].to_numpy(dtype=float)[order], color="crimson", lw=1.0)
    ax[2].plot(phase[order] + 1.0, df_base["baseline"].to_numpy(dtype=float)[order], color="crimson", lw=1.0, alpha=0.7)
    ax[2].set_title("Folded view")
    return ax


def select_example_trials(df: pd.DataFrame, mode: str = "phase_template_true_period") -> dict[str, int | None]:
    sub = df[(df["mode"] == mode) & df["status"].eq("ok")].copy()
    examples: dict[str, int | None] = {}
    hit = sub[sub["target_recovered"].fillna(False).astype(bool)]
    miss = sub[
        sub["has_dip"].fillna(False).astype(bool)
        & sub["truth_observable_actual"].fillna(False).astype(bool)
        & ~sub["target_recovered"].fillna(False).astype(bool)
    ]
    off_target = sub[sub["off_target_detection"].fillna(False).astype(bool)]
    control_fp = sub[sub["false_positive"].fillna(False).astype(bool)]
    fallback = sub[sub["phase_template_fallback"].fillna(False).astype(bool)]

    examples["strong_hit"] = int(hit.sort_values("dip_bayes_factor", ascending=False)["trial_id"].iloc[0]) if not hit.empty else None
    examples["observable_miss"] = int(miss.sort_values("dip_amp_mag", ascending=False)["trial_id"].iloc[0]) if not miss.empty else None
    examples["off_target_detection"] = int(off_target.sort_values("dip_bayes_factor", ascending=False)["trial_id"].iloc[0]) if not off_target.empty else None
    examples["control_false_positive"] = int(control_fp.sort_values("dip_bayes_factor", ascending=False)["trial_id"].iloc[0]) if not control_fp.empty else None
    examples["phase_template_fallback"] = int(fallback["trial_id"].iloc[0]) if not fallback.empty else None
    return examples
