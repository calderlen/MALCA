from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from malca.stv.events import score_lightcurve
from malca.core.baseline import (
    global_median_baseline,
    per_camera_gp_baseline,
    per_camera_gp_baseline_masked,
    per_camera_median_baseline,
)
from malca.core.utils import read_lc_dat2
from malca.config import (
    LOGBF_THRESHOLD_DIP,
    LOGBF_THRESHOLD_JUMP,
    SIGNIFICANCE_THRESHOLD,
    P_POINTS,
    MAG_POINTS,
    RUN_MIN_POINTS,
    RUN_MAX_GAP_POINTS,
    BASELINE_FUNC,
    BASELINE_S0,
    BASELINE_W0,
    BASELINE_Q,
    BASELINE_JITTER,
    MIN_MAG_OFFSET,
    FP_TRIALS_PER_FAMILY,
    INJECTION_SEED,
    DEFAULT_OUTPUT_DIR,
    LIGHT_CURVE_FILE_EXTENSION,
)
from malca.io.table_io import read_parquet_table, write_parquet_table


FALSE_POSITIVE_SCHEMA_VERSION = 2
_MISSING_ID_VALUES = frozenset({"", "nan", "none", "null", "<na>"})


def _canonical_candidate_id(value: object) -> str | None:
    if value is None or value is pd.NA:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, np.integer)):
        text = str(int(value))
    elif isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        text = str(int(value))
    else:
        text = str(value).strip()
    return None if text.lower() in _MISSING_ID_VALUES else text


def _fingerprintable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _fingerprintable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_fingerprintable(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, default=str)) if isinstance(value, (set, frozenset)) else items
    if callable(value):
        return f"{value.__module__}.{getattr(value, '__qualname__', getattr(value, '__name__', type(value).__name__))}"
    if isinstance(value, Path):
        return str(value.expanduser())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _stable_digest(value: object) -> str:
    payload = json.dumps(_fingerprintable(value), sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _code_fingerprint(detection_kwargs: dict | None = None) -> str:
    callables = {"score_lightcurve": score_lightcurve, **CONTAMINANT_FUNCS}
    def collect(prefix: str, value: object) -> None:
        if callable(value):
            callables[prefix] = value
        elif isinstance(value, dict):
            for key, item in value.items():
                collect(f"{prefix}.{key}", item)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                collect(f"{prefix}[{index}]", item)

    collect("detection_kwargs", detection_kwargs or {})
    sources: dict[str, str] = {}
    for name, function in callables.items():
        try:
            sources[name] = inspect.getsource(function)
        except (OSError, TypeError):
            sources[name] = repr(function)
    return _stable_digest(sources)


def _path_fingerprint(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve(strict=False)
    record: dict[str, object] = {"path": str(resolved), "exists": resolved.exists()}
    if not resolved.exists() or not resolved.is_file():
        return record
    stat = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    record.update({"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": digest.hexdigest()})
    return record


def _wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if trials <= 0:
        return None, None
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half_width = z * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials**2)) / denominator
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def _require_bool(value: object, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise ValueError(f"Detection result {field!r} must be an explicit boolean, got {value!r}")


def _build_detection_kwargs(trigger_mode: str) -> dict:
    baseline_map = {
        "gp": per_camera_gp_baseline,
        "gp_masked": per_camera_gp_baseline_masked,
        "global_median": global_median_baseline,
        "per_camera_median": per_camera_median_baseline,
    }
    if BASELINE_FUNC not in baseline_map:
        raise ValueError(f"Unsupported configured baseline {BASELINE_FUNC!r}")
    return dict(
        trigger_mode=trigger_mode,
        logbf_threshold_dip=LOGBF_THRESHOLD_DIP,
        logbf_threshold_jump=LOGBF_THRESHOLD_JUMP,
        significance_threshold=SIGNIFICANCE_THRESHOLD,
        p_points=P_POINTS,
        mag_points=MAG_POINTS,
        run_min_points=RUN_MIN_POINTS,
        max_gap_points=RUN_MAX_GAP_POINTS,
        run_max_gap_days=None,
        run_min_duration_days=None,
        compute_event_prob=True,
        baseline_func=baseline_map[BASELINE_FUNC],
        baseline_kwargs=dict(
            S0=BASELINE_S0,
            w0=BASELINE_W0,
            q=BASELINE_Q,
            jitter=BASELINE_JITTER,
            add_sigma_eff_col=True,
        ),
        # Consumed by _default_detection_func after scoring; score_lightcurve
        # itself intentionally does not own this post-score pipeline gate.
        min_mag_offset=MIN_MAG_OFFSET,
    )


def _default_detection_func(df: pd.DataFrame, detection_kwargs: dict) -> dict:
    score_kwargs = dict(detection_kwargs)
    min_mag_offset = float(score_kwargs.pop("min_mag_offset", 0.0) or 0.0)
    if "v_g_band" in df.columns:
        numeric_band = pd.to_numeric(df["v_g_band"], errors="coerce")
        band_frames = [
            ("g", df.loc[numeric_band.eq(0)].copy()),
            ("v", df.loc[numeric_band.eq(1)].copy()),
        ]
        band_frames = [(name, frame) for name, frame in band_frames if not frame.empty]
    else:
        band_frames = [("combined", df)]
    if not band_frames:
        raise ValueError("No g- or V-band observations are available for scoring")

    output: dict[str, object] = {}
    dip_decisions: list[bool] = []
    jump_decisions: list[bool] = []
    dip_triggers: list[float] = []
    jump_triggers: list[float] = []
    for band_name, band_frame in band_frames:
        res = score_lightcurve(band_frame, **score_kwargs)
        if not isinstance(res, dict) or not isinstance(res.get("dip"), dict) or not isinstance(res.get("jump"), dict):
            raise ValueError(f"score_lightcurve returned an incomplete dip/jump result for {band_name} band")
        dip = res["dip"]
        jump = res["jump"]
        dip_significant = _require_bool(dip.get("significant"), f"{band_name}.dip.significant")
        jump_significant = _require_bool(jump.get("significant"), f"{band_name}.jump.significant")

        def amplitude_pass(block: dict, significant: bool) -> bool:
            if not significant or min_mag_offset <= 0:
                return significant
            delta = block.get("best_delta_mag", block.get("best_mag_event", np.nan))
            try:
                value = float(delta)
            except (TypeError, ValueError):
                return False
            return bool(np.isfinite(value) and abs(value) >= min_mag_offset)

        dip_significant = amplitude_pass(dip, dip_significant)
        jump_significant = amplitude_pass(jump, jump_significant)
        dip_trigger = float(dip.get("trigger_max", np.nan))
        jump_trigger = float(jump.get("trigger_max", np.nan))
        dip_decisions.append(dip_significant)
        jump_decisions.append(jump_significant)
        dip_triggers.append(dip_trigger)
        jump_triggers.append(jump_trigger)
        output.update(
            {
                f"{band_name}_dip_significant": dip_significant,
                f"{band_name}_jump_significant": jump_significant,
                f"{band_name}_dip_trigger_max": dip_trigger,
                f"{band_name}_jump_trigger_max": jump_trigger,
            }
        )

    def finite_max(values: list[float]) -> float:
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        return float(np.max(finite)) if finite.size else np.nan

    output.update(
        {
            # The benchmark measures false dip triggers. Jump decisions remain
            # separate and are never silently folded into this numerator.
            "detected": bool(any(dip_decisions)),
            "dip_significant": bool(any(dip_decisions)),
            "jump_significant": bool(any(jump_decisions)),
            "dip_trigger_max": finite_max(dip_triggers),
            "jump_trigger_max": finite_max(jump_triggers),
            "bands_scored": ",".join(name for name, _ in band_frames),
        }
    )
    return output


def _inject_camera_offset(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    cams = out["camera#"].dropna().unique().tolist() if "camera#" in out.columns else []
    if not cams:
        return out
    bad_cam = rng.choice(cams)
    out.loc[out["camera#"] == bad_cam, "mag"] += rng.uniform(0.05, 0.35)
    return out


def _inject_semiregular(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    t = out["JD"].to_numpy(dtype=float)
    period = rng.uniform(20.0, 180.0)
    amp = rng.uniform(0.05, 0.4)
    out["mag"] = out["mag"].to_numpy(dtype=float) + amp * np.sin(2.0 * np.pi * (t - t.min()) / period)
    return out


def _inject_rcb_like(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    t = out["JD"].to_numpy(dtype=float)
    if len(t) == 0:
        return out
    center = rng.uniform(t.min(), t.max())
    width = rng.uniform(30.0, 250.0)
    depth = rng.uniform(0.4, 2.0)
    profile = np.exp(-0.5 * ((t - center) / max(width, 1e-3)) ** 2)
    out["mag"] = out["mag"].to_numpy(dtype=float) + depth * profile
    return out


def _inject_camera_cluster(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    if n < 10:
        return out
    idx = rng.choice(np.arange(n), size=max(3, int(0.1 * n)), replace=False)
    out.loc[out.index[idx], "mag"] += rng.normal(0.0, 0.6, size=len(idx))
    return out


def _inject_cv_outburst(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    t = out["JD"].to_numpy(dtype=float)
    if len(t) == 0:
        return out
    t_0 = rng.uniform(t.min(), t.max())
    tau_rise = rng.uniform(0.5, 5.0)
    tau_decay = rng.uniform(5.0, 40.0)
    amp = rng.uniform(-6.0, -2.0)  # negative because brightening

    dt = t - t_0
    profile = np.where(
        dt >= 0,
        np.exp(-dt / tau_decay),
        np.exp(dt / tau_rise)
    )
    out["mag"] = out["mag"].to_numpy(dtype=float) + amp * profile
    return out


def _inject_drw_agn(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    t = out["JD"].to_numpy(dtype=float)
    if len(t) == 0:
        return out

    # Ensure time is sorted for autoregressive generation
    sort_idx = np.argsort(t)
    t_sorted = t[sort_idx]

    tau = rng.uniform(50.0, 600.0)
    sf_inf = rng.uniform(0.1, 1.0)

    s = np.zeros(len(t_sorted))
    s[0] = rng.normal(0, sf_inf / np.sqrt(2.0))

    for i in range(1, len(t_sorted)):
        dt = t_sorted[i] - t_sorted[i-1]
        decay = np.exp(-dt / tau)
        sigma_drive = (sf_inf / np.sqrt(2.0)) * np.sqrt(max(0.0, 1.0 - decay**2))
        s[i] = decay * s[i-1] + rng.normal(0, sigma_drive)

    # Unsort to match original dataframe index order
    unsort_idx = np.argsort(sort_idx)
    s_original_order = s[unsort_idx]

    out["mag"] = out["mag"].to_numpy(dtype=float) + s_original_order
    return out


def _inject_eclipsing_binary(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    t = out["JD"].to_numpy(dtype=float)
    if len(t) == 0:
        return out

    period = rng.uniform(0.5, 30.0)
    t0 = rng.uniform(0.0, period)

    depth_pri = rng.uniform(0.3, 1.5)
    width_pri = rng.uniform(0.02, 0.1)  # phase width

    depth_sec = rng.uniform(0.05, 0.5)
    width_sec = width_pri * rng.uniform(0.8, 1.2)

    phase = ((t - t0) % period) / period

    # Distance to primary eclipse at phase 0 or 1
    dist_pri = np.minimum(phase, 1.0 - phase)
    # Distance to secondary eclipse at phase 0.5
    dist_sec = np.abs(phase - 0.5)

    profile = depth_pri * np.exp(-0.5 * (dist_pri / width_pri)**2) + \
              depth_sec * np.exp(-0.5 * (dist_sec / width_sec)**2)

    out["mag"] = out["mag"].to_numpy(dtype=float) + profile
    return out


def _inject_contact_binary(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Simulates a contact (W UMa) or semi-detached binary using ellipsoidal variations.
    Modeled as continuous Fourier series rather than discrete Gaussians.
    """
    out = df.copy()
    t = out["JD"].to_numpy(dtype=float)
    if len(t) == 0:
        return out

    # Contact binaries have short periods
    period = rng.uniform(0.2, 2.0)
    t0 = rng.uniform(0.0, period)

    phase = ((t - t0) % period) / period

    # Ellipsoidal variation (causes the two minima per orbit)
    amp_ellips = rng.uniform(0.05, 0.4)
    # Asymmetry (difference in depth between primary and secondary minima)
    amp_asym = rng.uniform(0.0, 0.2)

    # Positive mag = fainter.
    # cos(4*pi*phase) is +1 at phase 0 and 0.5 (the two eclipses/minima) and -1 at 0.25, 0.75 (maxima)
    # cos(2*pi*phase) is +1 at phase 0 (making primary deeper) and -1 at 0.5 (making secondary shallower)
    profile = amp_ellips * np.cos(4.0 * np.pi * phase) + amp_asym * np.cos(2.0 * np.pi * phase)

    out["mag"] = out["mag"].to_numpy(dtype=float) + profile
    return out


CONTAMINANT_FUNCS = {
    "camera_offset": _inject_camera_offset,
    "camera_cluster": _inject_camera_cluster,
    "semiregular": _inject_semiregular,
    "rcb_like": _inject_rcb_like,
    "cv_outburst": _inject_cv_outburst,
    "drw_agn": _inject_drw_agn,
    "eclipsing_binary": _inject_eclipsing_binary,
    "contact_binary": _inject_contact_binary,
}


def _run_single_injection(
    family: str,
    asas_sn_id: str,
    lc_dir: str,
    file_ext: str,
    trial: int,
    detection_kwargs: dict,
    seed: int,
    base_seed: int | None = None,
    trial_id: str | None = None,
    input_path: str | None = None,
    input_fingerprint: str | None = None,
    config_fingerprint: str | None = None,
    run_fingerprint: str | None = None,
) -> dict:
    fn = CONTAMINANT_FUNCS.get(family)
    rng = np.random.default_rng(seed)
    base = {
        "family": family,
        "trial": int(trial),
        "trial_index": int(trial),
        "trial_id": trial_id or f"{family}:{int(trial):08d}",
        "candidate_id": str(asas_sn_id),
        "asas_sn_id": str(asas_sn_id),
        "input_path": str(input_path or lc_dir),
        "base_seed": int(base_seed if base_seed is not None else seed),
        "trial_seed": int(seed),
        "input_fingerprint": input_fingerprint,
        "config_fingerprint": config_fingerprint,
        "run_fingerprint": run_fingerprint,
    }
    try:
        if fn is None:
            raise ValueError(f"Unknown contaminant family {family!r}")
        exact_input = Path(str(input_path or lc_dir)).expanduser()
        if not exact_input.is_file():
            return {
                **base,
                "trial_status": "input_missing",
                "detected": pd.NA,
                "dip_significant": pd.NA,
                "jump_significant": pd.NA,
                "injection_applied": False,
                "error": f"Light-curve input is missing or is not a file: {exact_input}",
            }
        # Passing the exact selected file path prevents a trial from loading a
        # different same-named light curve from the parent directory.
        df_g, df_v = read_lc_dat2(asas_sn_id, str(exact_input), file_ext=file_ext)
        df_lc = pd.concat([df_g, df_v], ignore_index=True)
        if df_lc.empty:
            return {
                **base,
                "trial_status": "ineligible_empty_light_curve",
                "detected": pd.NA,
                "dip_significant": pd.NA,
                "jump_significant": pd.NA,
                "injection_applied": False,
                "error": None,
            }
        df_bad = fn(df_lc, rng)
        original_mag = pd.to_numeric(df_lc.get("mag"), errors="coerce").to_numpy(dtype=float)
        injected_mag = pd.to_numeric(df_bad.get("mag"), errors="coerce").to_numpy(dtype=float)
        if original_mag.shape != injected_mag.shape:
            raise ValueError("Contaminant injection changed the light-curve row count")
        magnitude_delta = injected_mag - original_mag
        finite_delta = magnitude_delta[np.isfinite(magnitude_delta)]
        injection_scale = float(np.max(np.abs(finite_delta))) if finite_delta.size else np.nan
        if not np.isfinite(injection_scale) or injection_scale <= 0.0:
            return {
                **base,
                "trial_status": "ineligible_injection_not_applied",
                "detected": pd.NA,
                "dip_significant": pd.NA,
                "jump_significant": pd.NA,
                "injection_applied": False,
                "injection_max_abs_delta_mag": injection_scale,
                "error": None,
            }
        det = _default_detection_func(df_bad, detection_kwargs=detection_kwargs)
        band_provenance = {
            key: value
            for key, value in det.items()
            if key.startswith(("g_", "v_", "combined_")) or key == "bands_scored"
        }
        return {
            **base,
            **band_provenance,
            "trial_status": "ok",
            "detected": _require_bool(det.get("detected"), "detected"),
            "dip_significant": _require_bool(det.get("dip_significant"), "dip_significant"),
            "jump_significant": _require_bool(det.get("jump_significant"), "jump_significant"),
            "dip_trigger_max": float(det.get("dip_trigger_max", np.nan)),
            "jump_trigger_max": float(det.get("jump_trigger_max", np.nan)),
            "injection_applied": True,
            "injection_max_abs_delta_mag": injection_scale,
            "error": None,
        }
    except Exception as e:
        return {
            **base,
            "trial_status": "error",
            "detected": pd.NA,
            "dip_significant": pd.NA,
            "jump_significant": pd.NA,
            "injection_applied": pd.NA,
            "error": f"{type(e).__name__}: {e}",
        }


def _normalise_manifest(manifest_df: pd.DataFrame) -> pd.DataFrame:
    frame = manifest_df.copy()
    if "asas_sn_id" not in frame.columns:
        for alias in ("candidate_id", "source_id", "id"):
            if alias in frame.columns:
                frame["asas_sn_id"] = frame[alias]
                break
    if "path" not in frame.columns:
        for alias in ("dat_path", "lc_path"):
            if alias in frame.columns:
                frame["path"] = frame[alias]
                break
    if "asas_sn_id" not in frame.columns or "path" not in frame.columns:
        raise ValueError(f"manifest_df must contain a candidate identity and exact light-curve path, got {list(frame.columns)}")
    frame["candidate_id"] = frame["asas_sn_id"].map(_canonical_candidate_id).astype("string")
    if bool(frame["candidate_id"].isna().any()):
        raise ValueError("False-positive manifest contains blank/null candidate identities")
    duplicates = frame["candidate_id"].duplicated(keep=False)
    if bool(duplicates.any()):
        examples = sorted(frame.loc[duplicates, "candidate_id"].astype(str).unique())[:5]
        raise ValueError(f"False-positive manifest contains duplicate candidate identities: {examples}")
    raw_paths = frame["path"].map(lambda value: None if _canonical_candidate_id(value) is None else str(Path(str(value)).expanduser()))
    if bool(raw_paths.isna().any()):
        raise ValueError("False-positive manifest contains blank/null light-curve paths")
    exact_paths: list[str] = []
    for candidate_id, raw_path in zip(frame["candidate_id"].astype(str), raw_paths.astype(str)):
        path = Path(raw_path)
        if not path.is_file() and not path.suffix:
            path = path / f"{candidate_id}.{LIGHT_CURVE_FILE_EXTENSION}"
        exact_paths.append(str(path))
    frame["path"] = exact_paths
    duplicate_paths = frame["path"].map(lambda value: str(Path(value).resolve(strict=False))).duplicated(keep=False)
    if bool(duplicate_paths.any()):
        examples = sorted(frame.loc[duplicate_paths, "path"].astype(str).unique())[:5]
        raise ValueError(f"False-positive manifest assigns one light-curve path to multiple candidates: {examples}")
    frame["asas_sn_id"] = frame["candidate_id"]
    return frame.sort_values(["candidate_id", "path"], kind="mergesort").reset_index(drop=True)


def _manifest_fingerprint(frame: pd.DataFrame) -> tuple[str, dict[str, dict[str, object]]]:
    path_records: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for row in frame[["candidate_id", "path"]].itertuples(index=False):
        path_record = _path_fingerprint(Path(str(row.path)))
        path_records[str(row.candidate_id)] = path_record
        rows.append({"candidate_id": str(row.candidate_id), "path": str(row.path), "file": path_record})
    return _stable_digest(rows), path_records


def compute_false_positive_summary(
    trials: pd.DataFrame,
    *,
    families: list[str] | None = None,
    n_trials_per_family: int | None = None,
) -> pd.DataFrame:
    """Summarize designed trials without relabelling failures as negatives."""
    if "trial_id" in trials.columns and bool(trials["trial_id"].astype("string").duplicated().any()):
        raise ValueError("False-positive trials contain duplicate trial_id values")
    family_order = list(dict.fromkeys(families or trials.get("family", pd.Series(dtype=str)).dropna().astype(str)))
    rows: list[dict[str, object]] = []
    for family in family_order:
        group = trials.loc[trials.get("family", pd.Series(index=trials.index, dtype=str)).astype(str).eq(family)].copy()
        designed = int(n_trials_per_family) if n_trials_per_family is not None else int(len(group))
        if len(group) != designed:
            accounting_status = "trial_count_mismatch"
        else:
            accounting_status = "ok"
        status = group.get("trial_status", pd.Series("error", index=group.index)).astype("string")
        ok = status.eq("ok")
        errors = status.eq("error")
        ineligible = ~(ok | errors)
        detected = group.get("detected", pd.Series(pd.NA, index=group.index, dtype="boolean")).astype("boolean")
        if bool(detected.loc[ok].isna().any()):
            raise ValueError(f"Successful false-positive trials for {family!r} contain unknown decisions")
        if bool(detected.loc[~ok].notna().any()):
            raise ValueError(
                f"Non-successful false-positive trials for {family!r} contain detection decisions; "
                "errors/ineligible trials must remain unknown"
            )
        numerator = int(detected.loc[ok].sum())
        denominator = int(ok.sum())
        rate_low, rate_high = _wilson_interval(numerator, denominator)
        yield_low, yield_high = _wilson_interval(numerator, designed)
        rows.append(
            {
                "family": family,
                "count": denominator,  # Backward-compatible name: valid scientific denominator.
                "n_false_positive": numerator,
                "false_positive_rate": numerator / denominator if denominator else np.nan,
                "false_positive_rate_ci95_low": rate_low,
                "false_positive_rate_ci95_high": rate_high,
                "designed_trials": designed,
                "recorded_trials": int(len(group)),
                "successful_trials": denominator,
                "error_trials": int(errors.sum()),
                "ineligible_trials": int(ineligible.sum()),
                "nondetections": int(denominator - numerator),
                "unique_controls_evaluated": int(
                    group.loc[ok, "candidate_id"].nunique()
                    if "candidate_id" in group.columns
                    else group.loc[ok, "asas_sn_id"].nunique()
                    if "asas_sn_id" in group.columns
                    else 0
                ),
                "rate_numerator": numerator,
                "rate_denominator": denominator,
                "end_to_end_false_positive_yield": numerator / designed if designed else np.nan,
                "end_to_end_false_positive_yield_ci95_low": yield_low,
                "end_to_end_false_positive_yield_ci95_high": yield_high,
                "rate_denominator_definition": "trial_status == 'ok'",
                "accounting_status": accounting_status,
            }
        )
    return pd.DataFrame(rows)


def run_false_positive_benchmark(
    manifest_df: pd.DataFrame,
    *,
    out_dir: Path,
    families: list[str],
    n_trials_per_family: int,
    detection_kwargs: dict,
    seed: int,
    workers: int = 1,
    resume: bool = True,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if int(n_trials_per_family) < 0:
        raise ValueError("n_trials_per_family must be non-negative")
    if int(workers) < 1:
        raise ValueError("workers must be at least one")
    requested_families = list(dict.fromkeys(str(family).strip() for family in families if str(family).strip()))
    unknown = sorted(set(requested_families) - set(CONTAMINANT_FUNCS))
    if unknown:
        raise ValueError(f"Unknown contaminant families: {unknown}")
    frame = _normalise_manifest(manifest_df)
    if frame.empty and requested_families and int(n_trials_per_family) > 0:
        raise ValueError("False-positive manifest is empty")

    input_fingerprint, input_records = _manifest_fingerprint(frame)
    code_fingerprint = _code_fingerprint(detection_kwargs)
    config_fingerprint = _stable_digest(
        {
            "schema_version": FALSE_POSITIVE_SCHEMA_VERSION,
            "code_fingerprint": code_fingerprint,
            "families": requested_families,
            "n_trials_per_family": int(n_trials_per_family),
            "detection_kwargs": detection_kwargs,
            "seed": int(seed),
        }
    )
    run_fingerprint = _stable_digest(
        {
            "schema_version": FALSE_POSITIVE_SCHEMA_VERSION,
            "config_fingerprint": config_fingerprint,
            "input_fingerprint": input_fingerprint,
        }
    )

    tasks: list[tuple] = []
    for family in requested_families:
        for trial in range(int(n_trials_per_family)):
            # Each trial is independently derived from immutable identifiers;
            # adding a family or changing worker count cannot perturb old trials.
            selection_seed = int.from_bytes(
                hashlib.sha256(f"{int(seed)}|{family}|{int(trial)}|select".encode()).digest()[:8],
                "big",
            )
            trial_seed = int.from_bytes(
                hashlib.sha256(f"{int(seed)}|{family}|{int(trial)}|inject".encode()).digest()[:8],
                "big",
            )
            pick_index = int(np.random.default_rng(selection_seed).integers(0, len(frame)))
            pick = frame.iloc[pick_index]
            path = Path(str(pick["path"]))
            file_ext = path.suffix[1:] if path.suffix else "dat3"
            candidate_id = str(pick["candidate_id"])
            trial_id = f"{family}:{int(trial):08d}"
            candidate_input_fingerprint = _stable_digest(input_records[candidate_id])
            tasks.append(
                (
                    family,
                    candidate_id,
                    str(path),
                    file_ext,
                    trial,
                    detection_kwargs,
                    trial_seed,
                    int(seed),
                    trial_id,
                    str(path),
                    candidate_input_fingerprint,
                    config_fingerprint,
                    run_fingerprint,
                )
            )

    output_path = out_dir / "false_positive_trials.parquet"
    existing = pd.DataFrame()
    if resume and output_path.exists():
        candidate_existing = read_parquet_table(output_path)
        fingerprints = set(candidate_existing.get("run_fingerprint", pd.Series(dtype=str)).dropna().astype(str))
        if fingerprints == {run_fingerprint} and "trial_id" in candidate_existing.columns:
            if bool(candidate_existing["trial_id"].astype("string").duplicated().any()):
                raise ValueError("Existing false-positive output contains duplicate trial_id values")
            existing = candidate_existing.copy()
    completed = set(existing.get("trial_id", pd.Series(dtype=str)).dropna().astype(str))
    pending_tasks = [task for task in tasks if str(task[8]) not in completed]

    rows: list[dict] = existing.to_dict("records") if not existing.empty else []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_run_single_injection, *task): task for task in pending_tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="False Positive Trials"):
                task = futures[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    rows.append(
                        {
                            "family": task[0],
                            "trial": int(task[4]),
                            "trial_index": int(task[4]),
                            "trial_id": task[8],
                            "candidate_id": task[1],
                            "asas_sn_id": task[1],
                            "input_path": task[9],
                            "base_seed": int(seed),
                            "trial_seed": int(task[6]),
                            "input_fingerprint": task[10],
                            "config_fingerprint": config_fingerprint,
                            "run_fingerprint": run_fingerprint,
                            "trial_status": "error",
                            "detected": pd.NA,
                            "dip_significant": pd.NA,
                            "jump_significant": pd.NA,
                            "error": f"worker_error: {type(exc).__name__}: {exc}",
                        }
                    )
    else:
        for task in tqdm(pending_tasks, desc="False Positive Trials"):
            rows.append(_run_single_injection(*task))

    df = pd.DataFrame(rows)
    if not df.empty:
        if bool(df["trial_id"].astype("string").duplicated().any()):
            raise ValueError("False-positive output contains duplicate trial_id values")
        df = df.sort_values(["family", "trial"], kind="mergesort").reset_index(drop=True)
        expected_ids = {str(task[8]) for task in tasks}
        actual_ids = set(df["trial_id"].astype(str))
        if actual_ids != expected_ids:
            raise RuntimeError(
                "False-positive benchmark accounting mismatch: "
                f"missing={sorted(expected_ids - actual_ids)[:5]}, unexpected={sorted(actual_ids - expected_ids)[:5]}"
            )
        for column in ("detected", "dip_significant", "jump_significant"):
            if column in df.columns:
                df[column] = df[column].astype("boolean")

    summary = compute_false_positive_summary(
        df,
        families=requested_families,
        n_trials_per_family=int(n_trials_per_family),
    )
    write_parquet_table(df, output_path)
    write_parquet_table(summary, out_dir / "false_positive_summary.parquet")
    metadata = {
        "schema_version": FALSE_POSITIVE_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "config_fingerprint": config_fingerprint,
        "input_fingerprint": input_fingerprint,
        "code_fingerprint": code_fingerprint,
        "seed": int(seed),
        "families": requested_families,
        "n_trials_per_family": int(n_trials_per_family),
        "designed_trials": int(len(tasks)),
        "recorded_trials": int(len(df)),
    }
    metadata_path = out_dir / "false_positive_run.json"
    temp_metadata = metadata_path.with_suffix(".json.tmp")
    temp_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp_metadata, metadata_path)
    return df


def plot_false_alarm_survival(df: pd.DataFrame, out_dir: Path, trigger_mode: str = "posterior_prob"):
    """Plot False Alarm Rate vs Trigger Threshold."""
    col = "dip_trigger_max" if "dip_trigger_max" in df.columns else None
    if col is None:
        return
    status = df.get("trial_status", pd.Series("ok", index=df.index)).astype("string")
    successful = df.loc[status.eq("ok")].copy()
    if successful.empty:
        return
    plt.figure(figsize=(8, 6))
        
    # If using logbf, the range is 0 to ~20. If posterior_prob, 0 to 1.
    is_prob = (trigger_mode == "posterior_prob")
    t_min = 0.0
    t_max = 1.0 if is_prob else min(20.0, successful[col].max() if pd.notnull(successful[col].max()) else 10.0)
    thresholds = np.linspace(t_min, t_max, 100)
    
    for family, group in successful.groupby("family"):
        rates = []
        scores = pd.to_numeric(group[col], errors="coerce")
        if bool(scores.isna().any()):
            # A successful trial must have a score for every threshold; do not
            # hide malformed outputs by shrinking the numerator only.
            continue
        valid_scores = scores.to_numpy(dtype=float)
        n_total = len(group)
        if n_total == 0:
            continue
            
        for t in thresholds:
            n_trigger = np.sum(valid_scores >= t)
            rates.append(n_trigger / n_total)
            
        plt.plot(thresholds, rates, label=family, lw=2)
        
    plt.xlabel(f"Trigger Threshold ({trigger_mode})", fontsize=12)
    plt.ylabel("False Alarm Rate (FAR)", fontsize=12)
    plt.title("False Alarm Survival Curve", fontsize=14)
    plt.yscale("log")
    plt.ylim(bottom=1e-4, top=1.0)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / "false_alarm_survival.pdf")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="False-positive contaminant benchmark for MALCA")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_DIR / "lc_manifest_all.parquet", help="Manifest Parquet with asas_sn_id and path")
    parser.add_argument("--output-dir", dest="out_dir", type=Path, default=DEFAULT_OUTPUT_DIR / "false_positive")
    parser.add_argument("--families", type=str, default="camera_offset,camera_cluster,semiregular,rcb_like,cv_outburst,drw_agn,eclipsing_binary,contact_binary")
    parser.add_argument("--n-trials-per-family", type=int, default=FP_TRIALS_PER_FAMILY)
    parser.add_argument("--seed", type=int, default=INJECTION_SEED)
    parser.add_argument("--trigger-mode", type=str, default="posterior_prob", choices=["logbf", "posterior_prob"])
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--no-resume", action="store_true", help="Ignore compatible existing trial output and recompute all trials")
    args = parser.parse_args()

    df_manifest = read_parquet_table(args.manifest)
    
    # Handle schema from manifest builder
    if "source_id" in df_manifest.columns and "asas_sn_id" not in df_manifest.columns:
        df_manifest = df_manifest.rename(columns={"source_id": "asas_sn_id"})
    if "dat_path" in df_manifest.columns and "path" not in df_manifest.columns:
        df_manifest = df_manifest.rename(columns={"dat_path": "path"})
    elif "lc_dir" in df_manifest.columns and "path" not in df_manifest.columns:
        df_manifest = df_manifest.rename(columns={"lc_dir": "path"})

    detection_kwargs = _build_detection_kwargs(args.trigger_mode)

    df_results = run_false_positive_benchmark(
        df_manifest,
        out_dir=args.out_dir,
        families=[f.strip() for f in args.families.split(",") if f.strip()],
        n_trials_per_family=args.n_trials_per_family,
        detection_kwargs=detection_kwargs,
        seed=args.seed,
        workers=args.workers,
        resume=not args.no_resume,
    )
    
    plot_false_alarm_survival(df_results, args.out_dir, args.trigger_mode)
    print(f"Wrote false-positive benchmark outputs and plot to {args.out_dir}")


if __name__ == "__main__":
    main()
