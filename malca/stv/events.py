from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal, Mapping, TypeAlias
import argparse
import glob
import json
import multiprocessing as mp
import os
import os as _os
import sys
import traceback
import warnings

_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Prevent numba from spawning a thread pool in each worker process.
_os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from numba import njit, prange
from scipy.optimize import curve_fit
from scipy.special import logsumexp
from tqdm import tqdm
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

_trapezoid = getattr(np, "trapezoid", np.trapz)

from malca.baseline import (
    global_median_baseline,
    phase_template_baseline,
    per_camera_median_baseline,
    per_camera_gp_baseline,
    per_camera_gp_baseline_masked,
)
from malca.config import BAD_CAMERA_SCATTER_RATIO_THRESHOLD
from malca.config import PARQUET_OUTPUT_COMPRESSION, OUTPUT_FORMAT, EVENTS_OUTPUT_CHUNK_SIZE
from malca.config import LCV2_ROOT, DEFAULT_OUTPUT_DIR
from malca.config import (
    WORKERS, TRIGGER_MODE, P_POINTS, MAG_POINTS,
    LOGBF_THRESHOLD_DIP, LOGBF_THRESHOLD_JUMP, SIGNIFICANCE_THRESHOLD,
    MIN_MAG_OFFSET, RUN_MIN_POINTS, RUN_MAX_GAP_POINTS, MAG_BINS,
    BASELINE_FUNC, BASELINE_S0, BASELINE_W0, BASELINE_Q, BASELINE_JITTER,
)
from malca.cli_config import add_config_args, apply_config, namespace_keys
from malca.stv.score import compute_event_score
from malca.stats import log_gaussian, median_dt, bic
from malca.stv.triggering import resolve_trigger_indices
from malca.utils import (
    read_lc_dat2,
    read_lc_csv,
    read_skypatrol_csv,
    clean_lc,
    compute_field_summary,
    gaussian,
    paczynski_kernel,
    fred,
    skew_gaussian,
    filter_bad_cameras,
    filter_residual_bad_cameras,
    log as _log,
)

warnings.filterwarnings("ignore", message=".*Covariance of the parameters could not be estimated.*")
warnings.filterwarnings("ignore", message=".*overflow encountered in.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in.*", category=RuntimeWarning)



EventKind: TypeAlias = Literal["dip", "jump"]

DEFAULT_BASELINE_KWARGS = dict(
    S0=BASELINE_S0,
    w0=BASELINE_W0,
    q=BASELINE_Q,
    jitter=BASELINE_JITTER,
    sigma_floor=None,
    add_sigma_eff_col=True,
)

EVENTS_CONFIG_DEFAULTS = {
    "trigger_mode": TRIGGER_MODE,
    "logbf_threshold_dip": LOGBF_THRESHOLD_DIP,
    "logbf_threshold_jump": LOGBF_THRESHOLD_JUMP,
    "significance_threshold": SIGNIFICANCE_THRESHOLD,
    "p_points": P_POINTS,
    "mag_points": MAG_POINTS,
    "run_min_points": RUN_MIN_POINTS,
    "run_max_gap_points": RUN_MAX_GAP_POINTS,
    "run_max_gap_days": None,
    "run_min_duration_days": 0.0,
    "no_event_prob": False,
    "p_min_dip": None,
    "p_max_dip": None,
    "p_min_jump": None,
    "p_max_jump": None,
    "baseline_func": BASELINE_FUNC,
    "baseline_s0": BASELINE_S0,
    "baseline_w0": BASELINE_W0,
    "baseline_q": BASELINE_Q,
    "baseline_jitter": BASELINE_JITTER,
    "baseline_sigma_floor": None,
    "mag_min_dip": None,
    "mag_max_dip": None,
    "mag_min_jump": None,
    "mag_max_jump": None,
    "filter_bad_cameras": True,
    "bad_camera_scatter_ratio": BAD_CAMERA_SCATTER_RATIO_THRESHOLD,
    "min_mag_offset": MIN_MAG_OFFSET,
    "output_format": OUTPUT_FORMAT,
    "chunk_size": EVENTS_OUTPUT_CHUNK_SIZE,
    "max_error_fraction": 0.01,
}

EVENTS_CONFIG_PATH_KEYS = {"output", "error_output", "metadata", "input_file"}


EVENTS_CORE_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "timescale",
    "asas_sn_id",
    "lc_path",
    "dip_significant",
    "jump_significant",
    "n_points",
    "jd_first",
    "jd_last",
    "cadence_median_days",
    "dip_best_morph",
    "dip_best_delta_bic",
    "dip_best_width_param",
    "dip_symmetry_score",
    "dip_best_amp",
    "dip_best_t0",
    "dip_best_alpha",
    "dip_best_tau",
    "jump_best_morph",
    "jump_best_delta_bic",
    "jump_best_width_param",
    "jump_best_amp",
    "jump_best_t0",
    "jump_best_alpha",
    "jump_best_tau",
    "dip_count",
    "jump_count",
    "dip_run_count",
    "jump_run_count",
    "dip_max_run_points",
    "jump_max_run_points",
    "dip_max_run_duration",
    "jump_max_run_duration",
    "dip_max_run_sum",
    "jump_max_run_sum",
    "dip_max_run_max",
    "jump_max_run_max",
    "dip_max_run_cameras",
    "jump_max_run_cameras",
    "dip_max_log_bf_local",
    "jump_max_log_bf_local",
    "dip_bayes_factor",
    "jump_bayes_factor",
    "baseline_mag",
    "dip_best_p",
    "jump_best_p",
    "dip_best_mag_event",
    "jump_best_mag_event",
    "dip_trigger_max",
    "jump_trigger_max",
    "dip_max_event_prob",
    "jump_max_event_prob",
    "n_cameras",
    "camera_ids",
    "camera_min_points",
    "camera_max_points",
    "asassn_field_key",
    "asassn_fields",
    "asassn_field_count",
    "asassn_field_key_fraction",
    "camera_field_key",
    "camera_fields",
    "camera_field_count",
    "camera_field_key_fraction",
    "dipper_score",
    "dipper_n_dips",
    "dipper_n_valid_dips",
    "jumper_score",
    "jumper_n_jumps",
    "jumper_n_valid_jumps",
    "baseline_source",
    "trigger_mode",
    "dip_trigger_threshold",
    "jump_trigger_threshold",
    "bad_cameras_filtered",
    "dip_is_single_event",
    "dip_inter_event_spacing_median",
    "dip_inter_event_spacing_std",
    "dip_amplitude_consistency",
    "dip_duration_consistency",
    "jump_is_single_event",
    "jump_inter_event_spacing_median",
    "jump_inter_event_spacing_std",
    "jump_amplitude_consistency",
    "jump_duration_consistency",
)

EVENTS_KNOWN_METADATA_COLUMNS: tuple[str, ...] = (
    "vsx_sep_arcsec",
    "vsx_class",
    "excluded_cameras",
    "raw_median_suspect_cameras",
    "pre_periodicity_label",
    "pre_periodic_flag",
    "pre_periodicity_selected_period",
    "pre_periodicity_method",
)

EVENTS_EXTRA_COLUMNS: tuple[str, ...] = ("failed_signal_amplitude", "extra_json")

EVENTS_BOOL_COLUMNS: frozenset[str] = frozenset(
    {
        "dip_significant",
        "jump_significant",
        "dip_is_single_event",
        "jump_is_single_event",
        "failed_signal_amplitude",
        "pre_periodic_flag",
    }
)

EVENTS_INT_COLUMNS: frozenset[str] = frozenset(
    {
        "n_points",
        "dip_count",
        "jump_count",
        "dip_run_count",
        "jump_run_count",
        "dip_max_run_points",
        "jump_max_run_points",
        "dip_max_run_cameras",
        "jump_max_run_cameras",
        "n_cameras",
        "camera_min_points",
        "camera_max_points",
        "asassn_field_count",
        "camera_field_count",
        "dipper_n_dips",
        "dipper_n_valid_dips",
        "jumper_n_jumps",
        "jumper_n_valid_jumps",
    }
)

EVENTS_STRING_COLUMNS: frozenset[str] = frozenset(
    {
        "candidate_id",
        "timescale",
        "asas_sn_id",
        "lc_path",
        "dip_best_morph",
        "jump_best_morph",
        "camera_ids",
        "asassn_field_key",
        "asassn_fields",
        "camera_field_key",
        "camera_fields",
        "baseline_source",
        "trigger_mode",
        "bad_cameras_filtered",
        "vsx_class",
        "excluded_cameras",
        "raw_median_suspect_cameras",
        "pre_periodicity_label",
        "pre_periodicity_method",
        "extra_json",
    }
)

EVENTS_FLOAT_COLUMNS: frozenset[str] = frozenset(
    (
        set(EVENTS_CORE_COLUMNS)
        | set(EVENTS_KNOWN_METADATA_COLUMNS)
        | set(EVENTS_EXTRA_COLUMNS)
    )
    - set(EVENTS_BOOL_COLUMNS)
    - set(EVENTS_INT_COLUMNS)
    - set(EVENTS_STRING_COLUMNS)
)


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def build_events_writer_columns(metadata_columns: Iterable[str] | None = None) -> list[str]:
    metadata_source = [] if metadata_columns is None else metadata_columns
    metadata = [
        str(col)
        for col in metadata_source
        if str(col) not in {"", "lc_path", "extra_json"}
    ]
    return _unique_preserve_order(
        (
            *EVENTS_CORE_COLUMNS,
            "failed_signal_amplitude",
            *EVENTS_KNOWN_METADATA_COLUMNS,
            *metadata,
            "extra_json",
        )
    )


def _is_missing_value(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _json_safe_value(value: object) -> object:
    if _is_missing_value(value):
        return None
    if isinstance(value, np.generic):
        return _json_safe_value(value.item())
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _parse_extra_json(value: object) -> dict[str, object]:
    if _is_missing_value(value) or str(value).strip() == "":
        return {}
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {"value": _json_safe_value(value)}
    return parsed if isinstance(parsed, dict) else {"value": _json_safe_value(value)}


def _extra_json_series(df: pd.DataFrame, extra_columns: list[str]) -> pd.Series:
    if "extra_json" in df.columns:
        base_values = df["extra_json"].tolist()
    else:
        base_values = [None] * len(df)

    encoded: list[str] = []
    for row_idx, base_value in enumerate(base_values):
        payload = _parse_extra_json(base_value)
        for column in extra_columns:
            value = df.iloc[row_idx][column]
            if _is_missing_value(value):
                continue
            payload[str(column)] = _json_safe_value(value)
        encoded.append(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return pd.Series(encoded, index=df.index, dtype="string")


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    true_values = {"1", "true", "t", "yes", "y"}
    false_values = {"0", "false", "f", "no", "n"}
    values: list[object] = []
    for value in series.tolist():
        if _is_missing_value(value):
            values.append(pd.NA)
        elif isinstance(value, (bool, np.bool_)):
            values.append(bool(value))
        elif isinstance(value, (int, np.integer, float, np.floating)) and np.isfinite(value):
            values.append(bool(value))
        elif isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in true_values:
                values.append(True)
            elif lowered in false_values:
                values.append(False)
            else:
                values.append(pd.NA)
        else:
            values.append(pd.NA)
    return pd.Series(values, index=series.index, dtype="boolean")


def _coerce_int_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    try:
        return numeric.astype("Int64")
    except (TypeError, ValueError):
        return numeric.round().astype("Int64")


def _coerce_string_series(series: pd.Series) -> pd.Series:
    out = series.astype("string")
    out = out.mask(series.map(_is_missing_value), pd.NA)
    return out


def _column_kind(column: str, column_kinds: Mapping[str, str] | None = None) -> str:
    if column_kinds and column in column_kinds:
        return str(column_kinds[column])
    if column in EVENTS_BOOL_COLUMNS:
        return "bool"
    if column in EVENTS_INT_COLUMNS:
        return "int"
    if column in EVENTS_FLOAT_COLUMNS:
        return "float"
    return "string"


def infer_events_metadata_column_kinds(meta_df: pd.DataFrame | None) -> dict[str, str]:
    if meta_df is None:
        return {}
    kinds: dict[str, str] = {}
    for column in meta_df.columns:
        if column == "lc_path":
            continue
        if column in EVENTS_BOOL_COLUMNS:
            kinds[column] = "bool"
        elif column in EVENTS_INT_COLUMNS:
            kinds[column] = "int"
        elif column in EVENTS_FLOAT_COLUMNS:
            kinds[column] = "float"
        elif column in EVENTS_STRING_COLUMNS:
            kinds[column] = "string"
        elif pd.api.types.is_bool_dtype(meta_df[column]):
            kinds[column] = "bool"
        elif pd.api.types.is_integer_dtype(meta_df[column]):
            kinds[column] = "int"
        elif pd.api.types.is_float_dtype(meta_df[column]):
            kinds[column] = "float"
        else:
            kinds[column] = "string"
    return kinds


def normalize_events_frame(
    df: pd.DataFrame,
    schema_columns: Iterable[str],
    *,
    column_kinds: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    schema = _unique_preserve_order(schema_columns)
    if "extra_json" not in schema:
        schema.append("extra_json")
    out = df.copy()

    extra_columns = [col for col in out.columns if col not in schema]
    out["extra_json"] = _extra_json_series(out, extra_columns)
    if extra_columns:
        out = out.drop(columns=extra_columns)

    for column in schema:
        if column not in out.columns:
            out[column] = pd.NA
        kind = _column_kind(column, column_kinds)
        if kind == "bool":
            out[column] = _coerce_bool_series(out[column])
        elif kind == "int":
            out[column] = _coerce_int_series(out[column])
        elif kind == "float":
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
        else:
            out[column] = _coerce_string_series(out[column])
    return out.loc[:, schema]


def events_arrow_schema(
    schema_columns: Iterable[str],
    *,
    column_kinds: Mapping[str, str] | None = None,
) -> pa.Schema:
    fields: list[pa.Field] = []
    schema = _unique_preserve_order(schema_columns)
    if "extra_json" not in schema:
        schema.append("extra_json")
    for column in schema:
        kind = _column_kind(column, column_kinds)
        if kind == "bool":
            arrow_type = pa.bool_()
        elif kind == "int":
            arrow_type = pa.int64()
        elif kind == "float":
            arrow_type = pa.float64()
        else:
            arrow_type = pa.string()
        fields.append(pa.field(column, arrow_type))
    return pa.schema(fields)


def events_table_from_frame(
    df: pd.DataFrame,
    schema_columns: Iterable[str],
    *,
    column_kinds: Mapping[str, str] | None = None,
) -> pa.Table:
    normalized = normalize_events_frame(df, schema_columns, column_kinds=column_kinds)
    return pa.Table.from_pandas(
        normalized,
        schema=events_arrow_schema(normalized.columns, column_kinds=column_kinds),
        preserve_index=False,
    )


def sigmoid_spaced_p_grid(p_min=1e-4, p_max=1.0 - 1e-4, n=12):
    """
    Probability grid that is uniform in logit/sigmoid space.

    This corresponds to placing equal spacing in log-odds:
      q = log(p / (1 - p))
    then mapping back through the sigmoid.
    """
    p_min = float(np.clip(p_min, 1e-12, 1 - 1e-12))
    p_max = float(np.clip(p_max, 1e-12, 1 - 1e-12))
    q_min = np.log(p_min / (1.0 - p_min))
    q_max = np.log(p_max / (1.0 - p_max))
    q = np.linspace(q_min, q_max, int(n))
    return 1.0 / (1.0 + np.exp(-q))


def uniform_p_grid(p_min=0.9, p_max=1.0 - 1e-6, n=36):
    """
    Uniform grid in p for approximating int(dp) with a uniform prior P(p)=const.
    """
    p_min = float(np.clip(p_min, 1e-12, 1.0 - 1e-12))
    p_max = float(np.clip(p_max, 1e-12, 1.0 - 1e-12))
    if not (p_min < p_max):
        raise ValueError(f"Require p_min < p_max, got {p_min=} {p_max=}")
    return np.linspace(p_min, p_max, int(n), dtype=float)



def default_mag_grid(
    resid: np.ndarray,
    kind: EventKind,  # "dip" or "jump"
    n: int = 12,
):
    """
    
    """
    resid_finite = resid[np.isfinite(resid)]
    if len(resid_finite) == 0:
        raise ValueError("No finite residual values for grid construction")
    lo, hi = np.nanpercentile(resid, [5, 95])
    if not (np.isfinite(lo) and np.isfinite(hi)):
        med = np.nanmedian(resid)
        lo, hi = med - 0.5, med + 0.5
    spread = max(hi - lo, 0.05)

    if kind == "dip":
        start = 0.02
        stop = max(0.02, hi + 0.5 * spread)
    elif kind == "jump":
        start = min(-0.02, lo - 0.5 * spread)
        stop = -0.02
    else:
        raise ValueError("kind must be 'dip' or 'jump'")

    if start == stop:
        if kind == "dip":
            stop = start + 0.1
        else:
            stop = start - 0.1

    return np.linspace(start, stop, int(n))


def compute_symmetry_score(
    jd: np.ndarray,  # times (JD)
    resid: np.ndarray,  # mag - baseline (positive in dips)
    sigma_eff: np.ndarray, # mag uncertainties
    center_idx: int,  # run center index
    start_idx: int,  # run start index
    end_idx: int,  # run end index
) -> float:
    """Tzanidakis+2025 Eq. 5 symmetry score (ingress vs egress area)."""
    jd = np.asarray(jd, float)
    resid = np.asarray(resid, float)
    sigma_eff = np.asarray(sigma_eff, float)

    if not (0 <= start_idx < center_idx < end_idx < len(jd)):
        return np.nan

    # ingress segment [start..center], egress segment [center..end]
    t_ingress = jd[start_idx:center_idx + 1]
    resid_ingress = resid[start_idx:center_idx + 1]
    err_ingress = sigma_eff[start_idx:center_idx + 1]

    t_egress = jd[center_idx:end_idx + 1]
    resid_egress = resid[center_idx:end_idx + 1]
    err_egress = sigma_eff[center_idx:end_idx + 1]

    I_ingress = _trapezoid(resid_ingress, t_ingress)
    I_egress = _trapezoid(resid_egress, t_egress)

    # trapezoidal rule weights for variance calculation
    def trapezoid_weights(t):
        w = np.zeros_like(t)
        if len(t) > 1:
            w[0] = (t[1] - t[0]) / 2.0
            w[-1] = (t[-1] - t[-2]) / 2.0
            if len(t) > 2:
                w[1:-1] = (t[2:] - t[:-2]) / 2.0
        return w

    var_ingress = np.sum((trapezoid_weights(t_ingress) * err_ingress)**2)
    var_egress = np.sum((trapezoid_weights(t_egress) * err_egress)**2)

    denominator = np.sqrt(var_ingress + var_egress)
    if denominator < 1e-10:
        return 0.0

    return float((I_ingress - I_egress) / denominator)


def classify_run_morphology(
    jd: np.ndarray,
    mag: np.ndarray,
    sigma_eff: np.ndarray,
    run_idx: np.ndarray,
    *,
    baseline: np.ndarray | None = None,
    kind: EventKind = "dip",
):
    """
    Fits gaussian / skew_gaussian / paczynski / fred / noise to a padded run segment.
    *baseline* – full-length baseline array (use baseline[slice] as baseline_guess).
    """
    pad = 5
    start_i = int(max(0, run_idx[0] - pad))
    end_i = int(min(len(jd), run_idx[-1] + pad + 1))

    t_padded = jd[start_i:end_i]
    mag_padded = mag[start_i:end_i]
    sigma_eff_padded = sigma_eff[start_i:end_i]

    # Use sliced GP baseline as guess when available; fall back to nanmedian
    if baseline is not None:
        baseline_guess = float(np.nanmedian(baseline[start_i:end_i]))
    else:
        baseline_guess = float(np.nanmedian(mag_padded))

    abs_diff = np.abs(mag_padded - baseline_guess)
    peak_local_idx = int(np.argmax(abs_diff))
    t0_guess = t_padded[peak_local_idx]
    amp_guess_raw = float(mag_padded[peak_local_idx] - baseline_guess)
    amp_guess_dip = abs(amp_guess_raw) if np.isfinite(amp_guess_raw) else 0.1
    amp_guess_jump = -abs(amp_guess_raw) if np.isfinite(amp_guess_raw) else -0.1
    sigma_guess = max((t_padded[-1] - t_padded[0]) / 4.0, 0.01)
    delta_bic_threshold = 10.0

    resid_null = mag_padded - baseline_guess
    bic_null = bic(resid_null, sigma_eff_padded, 1)

    best_bic = bic_null
    best_model = "noise"
    best_params = {}

    if kind == "dip":
        try:
            popt_g, _ = curve_fit(
                gaussian, t_padded, mag_padded,
                p0=[amp_guess_dip, t0_guess, sigma_guess, baseline_guess],
                sigma=sigma_eff_padded, maxfev=2000
            )
            resid_g = mag_padded - gaussian(t_padded, *popt_g)
            bic_g = bic(resid_g, sigma_eff_padded, 4)

            if (popt_g[0] > 0) and bic_g < (best_bic - delta_bic_threshold):
                best_bic = bic_g
                best_model = "gaussian"
                best_params = {
                    "amp": popt_g[0], "t0": popt_g[1],
                    "sigma": popt_g[2], "baseline": popt_g[3]
                }
        except Exception:
            pass

    # skew_gaussian for dips (asymmetric profiles)
    if kind == "dip":
        try:
            popt_sg, _ = curve_fit(
                skew_gaussian, t_padded, mag_padded,
                p0=[amp_guess_dip, t0_guess, sigma_guess, baseline_guess, 0.0],
                sigma=sigma_eff_padded, maxfev=3000,
                bounds=(
                    [-np.inf, t_padded[0], 1e-5, -np.inf, -10],
                    [np.inf, t_padded[-1], np.inf, np.inf, 10]
                )
            )
            resid_sg = mag_padded - skew_gaussian(t_padded, *popt_sg)
            bic_sg = bic(resid_sg, sigma_eff_padded, 5)

            if (popt_sg[0] > 0) and bic_sg < (best_bic - delta_bic_threshold):
                best_bic = bic_sg
                best_model = "skew_gaussian"
                best_params = {
                    "amp": popt_sg[0], "t0": popt_sg[1],
                    "sigma": popt_sg[2], "baseline": popt_sg[3],
                    "alpha": popt_sg[4]
                }
        except Exception:
            pass

    if kind == "jump":
        try:
            popt_p, _ = curve_fit(
                paczynski_kernel, t_padded, mag_padded,
                p0=[amp_guess_jump, t0_guess, sigma_guess, baseline_guess],
                sigma=sigma_eff_padded, maxfev=2000
            )
            resid_p = mag_padded - paczynski_kernel(t_padded, *popt_p)
            bic_p = bic(resid_p, sigma_eff_padded, 4)

            if (popt_p[0] < 0) and bic_p < (best_bic - delta_bic_threshold):
                best_bic = bic_p
                best_model = "paczynski"
                best_params = {
                    "amp": popt_p[0], "t0": popt_p[1],
                    "tE": popt_p[2], "baseline": popt_p[3]
                }
        except Exception:
            pass

        try:
            popt_f, _ = curve_fit(
                fred, t_padded, mag_padded,
                p0=[amp_guess_jump, t0_guess, 0.05, baseline_guess],
                sigma=sigma_eff_padded, maxfev=2000
            )
            if popt_f[0] < 0:
                resid_f = mag_padded - fred(t_padded, *popt_f)
                bic_f = bic(resid_f, sigma_eff_padded, 4)

                if bic_f < (best_bic - delta_bic_threshold):
                    best_bic = bic_f
                    best_model = "fred"
                    best_params = {
                        "amp": popt_f[0], "t0": popt_f[1],
                        "tau": popt_f[2], "baseline": popt_f[3]
                    }
        except Exception:
            pass

    return {
        "morphology": best_model,
        "bic": float(best_bic),
        "delta_bic_null": float(bic_null - best_bic),
        "params": best_params,
    }



def build_runs(
    trig_idx: np.ndarray,
    jd: np.ndarray,
    *,
    max_gap_points: int = 1,
    max_gap_days: float | None = None,
):
    """Build runs from clustered triggered points."""
    jd = np.asarray(jd, float)
    trig_idx = np.asarray(trig_idx, dtype=int)
    trig_idx = trig_idx[(trig_idx >= 0) & (trig_idx < jd.size)]
    if trig_idx.size == 0:
        return []

    trig_idx = np.unique(trig_idx)
    trig_idx.sort()

    if max_gap_days is None:
        # 99.73th percentile (3-sigma) of gaps between sorted data points
        dt = np.diff(np.sort(jd))
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size > 0:
            max_gap_days = float(np.nanpercentile(dt, 99.73))
        else:
            max_gap_days = 5.0
    max_gap_days = float(max_gap_days)

    max_index_step = int(max_gap_points) + 1

    runs = []
    current_run = [int(trig_idx[0])]
    for k in range(1, trig_idx.size):
        i_prev = current_run[-1]
        i = int(trig_idx[k])

        idx_step = i - i_prev
        dt = jd[i] - jd[i_prev]

        if (idx_step <= max_index_step) and np.isfinite(dt) and (dt <= max_gap_days):
            current_run.append(i)
        else:
            runs.append(np.asarray(current_run, dtype=int))
            current_run = [i]
    runs.append(np.asarray(current_run, dtype=int))
    return runs


def filter_runs(
    runs,
    jd: np.ndarray,
    point_significance: np.ndarray,
    *,
    min_points: int = 2,
    min_duration_days: float | None = None,
    per_point_threshold: float | None = None,
    cam_vec: np.ndarray | None = None,
):
    """Filter runs by minimum points, duration, and per-point threshold."""
    jd = np.asarray(jd, float)
    point_significance = np.asarray(point_significance, float)

    cad = median_dt(jd)
    if min_duration_days is None:
        if np.isfinite(cad):
            min_duration_days = max(2.0 * cad, 2.0)
        else:
            min_duration_days = 2.0
    min_duration_days = float(min_duration_days)

    kept = []
    summaries = []

    for r in runs:
        r = np.asarray(r, dtype=int)
        if r.size == 0:
            continue

        n = int(r.size)
        dur = float(jd[r[-1]] - jd[r[0]]) if n >= 2 else 0.0
        vals = point_significance[r]
        run_max = float(np.nanmax(vals)) if np.isfinite(vals).any() else np.nan
        run_sum = float(np.nansum(vals)) if np.isfinite(vals).any() else np.nan
        run_n_cameras = None
        if cam_vec is not None:
            cams = np.asarray(cam_vec[r])
            if cams.size:
                cams = cams[~pd.isna(cams)]
            run_n_cameras = int(np.unique(cams.astype(str)).size) if cams.size else 0

        ok = True
        if n < int(min_points):
            ok = False
        if dur < min_duration_days:
            ok = False
        if (per_point_threshold is not None) and (not (np.isfinite(run_max) and run_max >= float(per_point_threshold))):
            ok = False

        summaries.append(
            dict(
                start_idx=int(r[0]),
                end_idx=int(r[-1]),
                n_points=n,
                start_jd=float(jd[r[0]]),
                end_jd=float(jd[r[-1]]),
                duration_days=dur,
                run_max=run_max,
                run_sum=run_sum,
                run_n_cameras=run_n_cameras,
                kept=bool(ok),
            )
        )

        if ok:
            kept.append(r)

    return kept, summaries


def summarize_kept_runs(
    kept_runs,
    jd: np.ndarray,
    point_significance: np.ndarray,
    cam_vec: np.ndarray | None = None,
):
    jd = np.asarray(jd, float)
    point_significance = np.asarray(point_significance, float)

    if not kept_runs:
        return dict(
            n_runs=0,
            max_run_points=0,
            max_run_duration=np.nan,
            max_run_sum=np.nan,
            max_run_max=np.nan,
            max_run_cameras=0,
        )

    max_pts = 0
    max_dur = -np.inf
    max_sum = -np.inf
    max_max = -np.inf
    max_cams = 0

    for r in kept_runs:
        r = np.asarray(r, int)
        max_pts = max(max_pts, int(r.size))
        if r.size >= 2:
            max_dur = max(max_dur, float(jd[r[-1]] - jd[r[0]]))
        else:
            max_dur = max(max_dur, 0.0)

        vals = point_significance[r]
        if np.isfinite(vals).any():
            max_sum = max(max_sum, float(np.nansum(vals)))
            max_max = max(max_max, float(np.nanmax(vals)))
        if cam_vec is not None:
            cams = np.asarray(cam_vec[r])
            if cams.size:
                cams = cams[~pd.isna(cams)]
            run_n_cameras = int(np.unique(cams.astype(str)).size) if cams.size else 0
            max_cams = max(max_cams, run_n_cameras)

    return dict(
        n_runs=int(len(kept_runs)),
        max_run_points=int(max_pts),
        max_run_duration=float(max_dur) if np.isfinite(max_dur) else np.nan,
        max_run_sum=float(max_sum) if np.isfinite(max_sum) else np.nan,
        max_run_max=float(max_max) if np.isfinite(max_max) else np.nan,
        max_run_cameras=int(max_cams),
    )


def compute_recurrence_stats(run_summaries: list[dict]) -> dict:
    """Compute inter-event recurrence statistics from run summaries.

    Parameters
    ----------
    run_summaries : list[dict]
        Each dict must have ``start_jd``, ``end_jd``, ``duration_days``,
        and ``run_max`` (peak significance amplitude).

    Returns
    -------
    dict
        is_single_event, inter_event_spacing_median, inter_event_spacing_std,
        amplitude_consistency, duration_consistency.
    """
    empty = dict(
        is_single_event=True,
        inter_event_spacing_median=np.nan,
        inter_event_spacing_std=np.nan,
        amplitude_consistency=np.nan,
        duration_consistency=np.nan,
    )
    if not run_summaries or len(run_summaries) < 1:
        return empty

    if len(run_summaries) == 1:
        return empty

    # Sort by start_jd to ensure chronological order
    sorted_runs = sorted(run_summaries, key=lambda s: s.get("start_jd", 0.0))

    # Inter-event spacing: gap between end of one event and start of the next
    spacings = []
    for i in range(1, len(sorted_runs)):
        prev_end = sorted_runs[i - 1].get("end_jd", np.nan)
        cur_start = sorted_runs[i].get("start_jd", np.nan)
        if np.isfinite(prev_end) and np.isfinite(cur_start):
            spacings.append(cur_start - prev_end)

    spacings = np.asarray(spacings, float)
    spacing_median = float(np.nanmedian(spacings)) if spacings.size else np.nan
    spacing_std = float(np.nanstd(spacings, ddof=1)) if spacings.size >= 2 else np.nan

    # Amplitude consistency: coefficient of variation of run_max across runs
    amps = np.asarray([s.get("run_max", np.nan) for s in sorted_runs], float)
    amps = amps[np.isfinite(amps)]
    if amps.size >= 2 and np.mean(amps) != 0:
        amplitude_consistency = float(np.std(amps, ddof=1) / np.mean(amps))
    else:
        amplitude_consistency = np.nan

    # Duration consistency: coefficient of variation of duration_days
    durs = np.asarray([s.get("duration_days", np.nan) for s in sorted_runs], float)
    durs = durs[np.isfinite(durs)]
    if durs.size >= 2 and np.mean(durs) != 0:
        duration_consistency = float(np.std(durs, ddof=1) / np.mean(durs))
    else:
        duration_consistency = np.nan

    return dict(
        is_single_event=False,
        inter_event_spacing_median=spacing_median,
        inter_event_spacing_std=spacing_std,
        amplitude_consistency=amplitude_consistency,
        duration_consistency=duration_consistency,
    )


@njit(fastmath=True, cache=True, parallel=True)
def marginal_loglikelihood_grid(log_Pb, log_Pf, log_p, log_1_minus_p):
    """Marginal log-likelihood over the (mag_grid x p_grid) posterior grid."""
    M, N = log_Pb.shape  # shape: M x N
    P = log_p.shape[0]
    loglik = np.zeros((M, P), dtype=log_Pb.dtype)

    for m in prange(M):
        for p in range(P):
            lp = log_p[p]
            l1mp = log_1_minus_p[p]
            acc = 0.0

            for n in range(N):
                val_b = log_Pb[m, n] + lp
                val_f = log_Pf[m, n] + l1mp

                if val_b > val_f:
                    mix = val_b + np.log1p(np.exp(val_f - val_b))
                else:
                    mix = val_f + np.log1p(np.exp(val_b - val_f))

                acc += mix

            loglik[m, p] = acc

    return loglik

@njit(fastmath=True, cache=True, parallel=True)
def loo_event_probabilities(loglik, log_p, log_1_minus_p, log_Pb, log_Pf, is_faint):
    """Leave-one-out posterior event-probability for every data point."""
    M, P = loglik.shape
    _, N = log_Pb.shape  # shape: M x N

    event_prob = np.zeros(N, dtype=np.float64)

    for n in prange(N):
        max_b = -np.inf
        sum_b = 0.0

        max_f = -np.inf
        sum_f = 0.0

        for m in range(M):
            val_Pb = log_Pb[m, n]
            val_Pf = log_Pf[m, n]

            for p in range(P):
                t1 = log_p[p] + val_Pb
                t2 = log_1_minus_p[p] + val_Pf

                if t1 > t2:
                    mix = t1 + np.log1p(np.exp(t2 - t1))
                else:
                    mix = t2 + np.log1p(np.exp(t1 - t2))

                ll_excl = loglik[m, p] - mix

                val_b = ll_excl + t1
                val_f = ll_excl + t2

                if val_b > max_b:
                    sum_b = sum_b * np.exp(max_b - val_b) + 1.0
                    max_b = val_b
                else:
                    sum_b += np.exp(val_b - max_b)

                if val_f > max_f:
                    sum_f = sum_f * np.exp(max_f - val_f) + 1.0
                    max_f = val_f
                else:
                    sum_f += np.exp(val_f - max_f)

        log_bright = max_b + np.log(sum_b)
        log_faint = max_f + np.log(sum_f)

        if log_bright > log_faint:
            log_norm = log_bright + np.log1p(np.exp(log_faint - log_bright))
        else:
            log_norm = log_faint + np.log1p(np.exp(log_bright - log_faint))

        if is_faint:
            event_prob[n] = np.exp(log_faint - log_norm)
        else:
            event_prob[n] = np.exp(log_bright - log_norm)

    return event_prob



def score_events_bayesian(
    df: pd.DataFrame,
    *,
    kind: EventKind = "dip",
    mag_col: str = "mag",
    err_col: str = "error",

    baseline_func=per_camera_gp_baseline_masked,
    baseline_kwargs: dict | None = None,
    df_base: pd.DataFrame | None = None,

    p_min: float | None = None,
    p_max: float | None = None,
    p_points: int = 12,
    mag_grid: np.ndarray | None = None,
    mag_points: int = 12,

    trigger_mode: str = "posterior_prob",
    logbf_threshold: float = 5.0,
    significance_threshold: float = 99.99997,

    run_min_points: int = 2,
    max_gap_points: int = 1,
    run_max_gap_days: float | None = None,
    run_min_duration_days: float | None = None,

    compute_event_prob: bool = True,
):
    """
    Returns a dict including:
      - log_bf_local (N,)
      - event_probability (N,) if compute_event_prob
      - event_indices (after run gating)
      - significant (after run gating)
      - run diagnostics
      - global bayes_factor
    """
    # Only clean if df_base was not pre-computed; if df_base is provided,
    # the caller already cleaned df and df_base must match it.
    if df_base is None:
        df = clean_lc(df)
    cam_vec = df["camera#"].to_numpy() if "camera#" in df.columns else None
    jd = np.asarray(df["JD"], float)
    mags = np.asarray(df[mag_col], float)

    mags_finite = np.isfinite(mags).sum()
    mags_total = len(mags)
    if mags_finite == 0:
        raise ValueError(
            f"All magnitudes are NaN/inf after reading: "
            f"total={mags_total}, finite={mags_finite}, "
            f"NaN={np.isnan(mags).sum()}, inf={np.isinf(mags).sum()}"
        )

    raw_mag_err = np.asarray(df[err_col], float)

    raw_mag_err_finite = np.isfinite(raw_mag_err).sum()
    raw_mag_err_positive = (raw_mag_err > 0).sum() if raw_mag_err_finite > 0 else 0
    if raw_mag_err_finite == 0:
        raise ValueError(
            f"All errors are NaN/inf: "
            f"total={len(raw_mag_err)}, finite={raw_mag_err_finite}, "
            f"NaN={np.isnan(raw_mag_err).sum()}, inf={np.isinf(raw_mag_err).sum()}"
        )
    if raw_mag_err_positive == 0:
        raise ValueError(
            f"All errors are non-positive: "
            f"total={len(raw_mag_err)}, finite={raw_mag_err_finite}, positive={raw_mag_err_positive}, "
            f"min={np.nanmin(raw_mag_err) if raw_mag_err_finite > 0 else 'N/A'}"
        )

    sigma_eff = raw_mag_err.copy()

    if baseline_kwargs is None:
        baseline_kwargs = dict(DEFAULT_BASELINE_KWARGS)

    if df_base is None and baseline_func is not None:
        df_base = baseline_func(df, **baseline_kwargs)

    if df_base is None:
        if not np.isfinite(mags).any():
            raise ValueError("All magnitude values are NaN/inf")
        baseline_mags = np.full_like(mags, np.nanmedian(mags))
        baseline_sources = np.full(len(mags), "global_median", dtype=object)
    else:
        if "baseline" in df_base.columns:
            baseline_mags = np.asarray(df_base["baseline"], float)
        else:
            baseline_mags = np.asarray(df_base[mag_col], float)
        if "baseline_source" in df_base.columns:
            baseline_sources = np.asarray(df_base["baseline_source"], dtype=object)
        else:
            baseline_sources = np.full(len(df_base), "unknown", dtype=object)

        # sigma_eff is mandatory — every baseline must produce it
        if "sigma_eff" not in df_base.columns:
            raise RuntimeError("Baseline did not return 'sigma_eff'. All baselines must produce sigma_eff.")
        sigma_eff = np.asarray(df_base["sigma_eff"], float)
        sigma_eff_finite = np.isfinite(sigma_eff).sum()
        sigma_eff_positive = (sigma_eff > 0).sum() if sigma_eff_finite > 0 else 0
        if sigma_eff_finite == 0:
            raise ValueError(
                f"Baseline returned all NaN/inf sigma_eff: "
                f"total={len(sigma_eff)}, finite={sigma_eff_finite}, "
                f"NaN={np.isnan(sigma_eff).sum()}, inf={np.isinf(sigma_eff).sum()}"
            )
        if sigma_eff_positive == 0:
            raise ValueError(
                f"Baseline returned all non-positive sigma_eff: "
                f"total={len(sigma_eff)}, finite={sigma_eff_finite}, positive={sigma_eff_positive}, "
                f"min={np.nanmin(sigma_eff) if sigma_eff_finite > 0 else 'N/A'}"
            )

    baseline_finite = np.isfinite(baseline_mags).sum()
    if baseline_finite == 0:
        raise ValueError(
            f"Baseline function returned all NaN/inf values: "
            f"total={len(baseline_mags)}, finite={baseline_finite}, "
            f"NaN={np.isnan(baseline_mags).sum()}, inf={np.isinf(baseline_mags).sum()}, "
            f"baseline_func={baseline_func.__name__ if baseline_func else 'None'}"
        )
    
    sigma_eff_finite_final = np.isfinite(sigma_eff).sum()
    sigma_eff_positive_final = (sigma_eff > 0).sum() if sigma_eff_finite_final > 0 else 0
    if sigma_eff_finite_final == 0:
        raise ValueError(
            f"All errors are NaN/inf after baseline: "
            f"total={len(sigma_eff)}, finite={sigma_eff_finite_final}, "
            f"NaN={np.isnan(sigma_eff).sum()}, inf={np.isinf(sigma_eff).sum()}"
        )
    if sigma_eff_positive_final == 0:
        raise ValueError(
            f"All errors are non-positive after baseline: "
            f"total={len(sigma_eff)}, finite={sigma_eff_finite_final}, positive={sigma_eff_positive_final}, "
            f"min={np.nanmin(sigma_eff) if sigma_eff_finite_final > 0 else 'N/A'}"
        )
    
    total_points = len(mags)
    valid_mask = (
        np.isfinite(mags)
        & np.isfinite(sigma_eff)
        & (sigma_eff > 0)
        & np.isfinite(baseline_mags)
    )
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        raise ValueError(
            "No valid points after baseline/error filtering: "
            f"total={total_points}, finite_mags={np.isfinite(mags).sum()}, "
            f"finite_errs={np.isfinite(sigma_eff).sum()}, positive_errs={(sigma_eff > 0).sum()}, "
            f"finite_baseline={np.isfinite(baseline_mags).sum()}"
        )
    if n_valid < total_points:
        mags = mags[valid_mask]
        sigma_eff = sigma_eff[valid_mask]
        baseline_mags = baseline_mags[valid_mask]
        baseline_sources = baseline_sources[valid_mask]
        jd = jd[valid_mask]
        if cam_vec is not None:
            cam_vec = cam_vec[valid_mask]

    resid = mags - baseline_mags
    baseline_mag = float(np.nanmedian(baseline_mags))

    if kind == "dip":
        default_p_min, default_p_max = 0.5, 1.0 - 1e-4
    elif kind == "jump":
        default_p_min, default_p_max = 1e-4, 0.5
    else:
        raise ValueError("kind must be 'dip' or 'jump'")

    if p_min is None:
        p_min = default_p_min
    if p_max is None:
        p_max = default_p_max

    p_grid = uniform_p_grid(p_min=p_min, p_max=p_max, n=p_points)

    if mag_grid is None:
        mag_grid = default_mag_grid(resid, kind, n=mag_points)
    else:
        mag_grid = np.asarray(mag_grid, float)

    M = int(len(mag_grid))
    N = int(len(mags))

    if kind == "dip":
        log_Pb_vec = log_gaussian(resid, 0.0, sigma_eff)
        log_Pb_grid = np.broadcast_to(log_Pb_vec, (M, N))
        log_Pf_grid = log_gaussian(resid[None, :], mag_grid[:, None], sigma_eff)
        event_component = "faint"

    elif kind == "jump":
        log_Pb_grid = log_gaussian(resid[None, :], mag_grid[:, None], sigma_eff)
        log_Pf_vec = log_gaussian(resid, 0.0, sigma_eff)
        log_Pf_grid = np.broadcast_to(log_Pf_vec, (M, N))
        event_component = "bright"
        
        if not np.isfinite(log_Pf_vec).any():
            raise ValueError("All baseline likelihood values are NaN/inf")
        if not np.isfinite(log_Pb_grid).any():
            raise ValueError("All event likelihood values are NaN/inf")

    else:
        raise ValueError("kind must be 'dip' or 'jump'")

    valid_points = (np.isfinite(log_Pb_grid).any(axis=0)) | (np.isfinite(log_Pf_grid).any(axis=0))
    n_valid_points = int(valid_points.sum())
    total_points = log_Pb_grid.shape[1]
    if n_valid_points == 0:
        raise ValueError(
            "No valid likelihood contributions after baseline: "
            f"total={total_points}, baseline_finite={np.isfinite(log_Pb_grid).sum()}, "
            f"event_finite={np.isfinite(log_Pf_grid).sum()}"
        )
    if n_valid_points < total_points:
        mags = mags[valid_points]
        sigma_eff = sigma_eff[valid_points]
        baseline_mags = baseline_mags[valid_points]
        baseline_sources = baseline_sources[valid_points]
        jd = jd[valid_points]
        if cam_vec is not None:
            cam_vec = cam_vec[valid_points]
        log_Pb_grid = log_Pb_grid[:, valid_points]
        log_Pf_grid = log_Pf_grid[:, valid_points]
        if kind == "dip":
            log_Pb_vec = log_Pb_vec[valid_points]
        else:
            log_Pf_vec = log_Pf_vec[valid_points]
        N = n_valid_points

    if kind == "dip":
        loglik_baseline_only = float(np.sum(log_Pb_vec))
        log_px_baseline = log_Pb_vec
        log_px_event = logsumexp(log_Pf_grid, axis=0) - np.log(M)
    else:
        loglik_baseline_only = float(np.sum(log_Pf_vec))
        log_px_baseline = log_Pf_vec
        log_px_event = logsumexp(log_Pb_grid, axis=0) - np.log(M)

    log_bf_local = log_px_event - log_px_baseline

    max_log_bf_local = float(np.nanmax(log_bf_local)) if np.isfinite(log_bf_local).any() else np.nan

    log_p = np.log(p_grid)
    log_1_minus_p = np.log1p(-p_grid)

    loglik = marginal_loglikelihood_grid(
        np.ascontiguousarray(log_Pb_grid),
        np.ascontiguousarray(log_Pf_grid),
        log_p,
        log_1_minus_p
    )

    loglik_finite = np.isfinite(loglik).sum()
    loglik_total = loglik.size
    loglik_inf_neg = np.isinf(loglik) & (loglik < 0)
    loglik_inf_neg_count = loglik_inf_neg.sum()
    
    if loglik_finite == 0:
        if loglik_inf_neg_count == loglik_total:
            raise ValueError(
                f"All loglik values are -inf (all inputs were invalid): "
                f"total={loglik_total}, finite={loglik_finite}, -inf={loglik_inf_neg_count}, "
                f"This indicates all data points or baseline values were invalid."
            )
        else:
            raise ValueError(
                f"All loglik values are NaN/inf before normalization: "
                f"total={loglik_total}, finite={loglik_finite}, "
                f"NaN={np.isnan(loglik).sum()}, -inf={loglik_inf_neg_count}, +inf={np.isinf(loglik).sum() - loglik_inf_neg_count}"
            )
    
    loglik_sum = logsumexp(loglik)
    if not np.isfinite(loglik_sum):
        raise ValueError(
            f"logsumexp(loglik) is NaN/inf: "
            f"loglik_sum={loglik_sum}, loglik_finite={loglik_finite}/{loglik_total}, "
            f"loglik_min={np.nanmin(loglik) if loglik_finite > 0 else 'N/A'}, "
            f"loglik_max={np.nanmax(loglik) if loglik_finite > 0 else 'N/A'}"
        )
    
    log_post_norm = loglik - loglik_sum
    
    log_post_finite = np.isfinite(log_post_norm).sum()
    if log_post_finite == 0:
        raise ValueError(
            f"All log_posterior values are NaN/inf after normalization: "
            f"total={log_post_norm.size}, finite={log_post_finite}, "
            f"loglik_finite={loglik_finite}/{loglik_total}, loglik_sum={loglik_sum}, "
            f"loglik_range=[{np.nanmin(loglik) if loglik_finite > 0 else 'N/A'}, {np.nanmax(loglik) if loglik_finite > 0 else 'N/A'}]"
        )
    
    best_m_idx, best_p_idx = np.unravel_index(np.nanargmax(log_post_norm), log_post_norm.shape)
    best_mag_event = float(mag_grid[int(best_m_idx)])
    best_p = float(p_grid[int(best_p_idx)])

    K = loglik.size
    log_evidence_mixture = float(logsumexp(loglik) - np.log(K))
    bayes_factor = float(log_evidence_mixture - loglik_baseline_only)

    if compute_event_prob:
        event_prob = loo_event_probabilities(
                loglik,
                log_p,
                log_1_minus_p,
                log_Pb_grid,
                log_Pf_grid,
                (event_component == "faint")
            )
    else:
        event_prob = None

    trigger = resolve_trigger_indices(
        trigger_mode=trigger_mode,
        log_bf_local=log_bf_local,
        event_probability=event_prob,
        logbf_threshold=logbf_threshold,
        significance_threshold=significance_threshold,
    )
    point_significance = trigger["point_significance"]
    raw_idx = trigger["event_indices"]
    trigger_threshold_used = trigger["trigger_threshold"]
    trigger_value_max = trigger["trigger_max"]

    kept_runs = []
    run_summaries = []

    # Pull baseline array for morphology classification
    baseline_arr = np.asarray(df_base["baseline"], float) if (df_base is not None and "baseline" in df_base.columns) else None

    if raw_idx.size == 0:
        event_indices = np.array([], dtype=int)
        significant = False
        run_stats = summarize_kept_runs([], jd, point_significance, cam_vec=cam_vec)
    else:
        runs = build_runs(
            raw_idx,
            jd,
            max_gap_points=int(max_gap_points),
            max_gap_days=run_max_gap_days,
        )

        kept_runs, initial_summaries = filter_runs(
            runs,
            jd,
            point_significance,
            min_points=int(run_min_points),
            min_duration_days=run_min_duration_days,
            per_point_threshold=trigger_threshold_used,
            cam_vec=cam_vec,
        )

        kept_summaries = [s for s in initial_summaries if s.get("kept")]

        final_summaries = []
        for r, summary in zip(kept_runs, kept_summaries):
            summary = dict(summary)
            morph_res = classify_run_morphology(
                jd,
                mags,
                sigma_eff,
                r,
                baseline=baseline_arr,
                kind=kind,
            )
            summary.update(morph_res)
            
            # Symmetry score for dips (Tzanidakis+2025 Eq. 5), computed on residuals
            if kind == "dip" and len(r) >= 3:
                resid = mags - baseline_mags
                center_idx = int(r[np.argmax(resid[r])])
                start_idx = int(r[0])
                end_idx = int(r[-1])
                summary["symmetry_score"] = compute_symmetry_score(jd, resid, sigma_eff, center_idx, start_idx, end_idx)
            else:
                summary["symmetry_score"] = np.nan
            
            final_summaries.append(summary)
        
        run_summaries = final_summaries

        if kept_runs:
            event_indices = np.unique(np.concatenate(kept_runs)).astype(int)
            significant = True
        else:
            event_indices = np.array([], dtype=int)
            significant = False

        run_stats = summarize_kept_runs(kept_runs, jd, point_significance, cam_vec=cam_vec)

    return dict(
        kind=str(kind),
        baseline_mag=float(baseline_mag),
        best_mag_event=float(best_mag_event),
        best_p=float(best_p),

        log_bf_local=log_bf_local,
        max_log_bf_local=float(max_log_bf_local) if np.isfinite(max_log_bf_local) else np.nan,
        event_probability=event_prob,

        trigger_mode=str(trigger_mode),
        trigger_threshold=float(trigger_threshold_used),
        trigger_max=float(trigger_value_max) if np.isfinite(trigger_value_max) else np.nan,
        event_indices=event_indices,
        significant=bool(significant),

        run_summaries=run_summaries,
        **run_stats,

        bayes_factor=float(bayes_factor),
        log_evidence_mixture=float(log_evidence_mixture),
        log_evidence_baseline=float(loglik_baseline_only),
        baseline_source=",".join(sorted({str(x) for x in baseline_sources if isinstance(x, (str, bytes)) and len(str(x)) > 0})) or "unknown",

        p_grid=p_grid,
        mag_grid=mag_grid,

        df_base=df_base,
    )


def score_lightcurve(
    df: pd.DataFrame,
    *,
    baseline_func=per_camera_gp_baseline,
    baseline_kwargs: dict | None = None,
    filter_residual_bad_cameras_enabled: bool = False,
    bad_camera_scatter_ratio: float = BAD_CAMERA_SCATTER_RATIO_THRESHOLD,

    p_points: int = 12,
    mag_points: int = 12,
    trigger_mode: str = "posterior_prob",
    logbf_threshold_dip: float = 5.0,
    logbf_threshold_jump: float = 5.0,
    significance_threshold: float = 99.99997,

    run_min_points: int = 2,
    max_gap_points: int = 1,
    run_max_gap_days: float | None = None,
    run_min_duration_days: float | None = None,

    compute_event_prob: bool = True,

    p_min_dip: float | None = None,
    p_max_dip: float | None = None,
    p_min_jump: float | None = None,
    p_max_jump: float | None = None,
    mag_grid_dip: np.ndarray | None = None,
    mag_grid_jump: np.ndarray | None = None,
):
    """Compute baseline once, then score dips and jumps via kind_configs loop."""
    df = clean_lc(df)

    if baseline_kwargs is None:
        baseline_kwargs = dict(DEFAULT_BASELINE_KWARGS)

    df_base = baseline_func(df, **baseline_kwargs) if baseline_func is not None else None
    residual_bad_cameras: set = set()
    if filter_residual_bad_cameras_enabled and df_base is not None and "camera#" in df.columns:
        df_filtered, residual_bad_cameras = filter_residual_bad_cameras(
            df,
            df_base,
            scatter_ratio_threshold=bad_camera_scatter_ratio,
        )
        if residual_bad_cameras:
            df = df_filtered
            df_base = baseline_func(df, **baseline_kwargs) if baseline_func is not None else None

    kind_configs = {
        "dip": dict(
            p_min=p_min_dip, p_max=p_max_dip,
            mag_grid=mag_grid_dip, logbf_threshold=logbf_threshold_dip,
        ),
        "jump": dict(
            p_min=p_min_jump, p_max=p_max_jump,
            mag_grid=mag_grid_jump, logbf_threshold=logbf_threshold_jump,
        ),
    }

    results = {}
    for kind, cfg in kind_configs.items():
        results[kind] = score_events_bayesian(
            df, kind=kind,
            baseline_func=None, baseline_kwargs=baseline_kwargs, df_base=df_base,
            p_min=cfg["p_min"], p_max=cfg["p_max"],
            p_points=p_points, mag_grid=cfg["mag_grid"], mag_points=mag_points,
            trigger_mode=trigger_mode, logbf_threshold=cfg["logbf_threshold"],
            significance_threshold=significance_threshold,
            run_min_points=run_min_points, max_gap_points=max_gap_points,
            run_max_gap_days=run_max_gap_days,
            run_min_duration_days=run_min_duration_days,
            compute_event_prob=compute_event_prob,
        )

    return dict(
        dip=results["dip"],
        jump=results["jump"],
        df_base=df_base,
        df=df,
        bad_cameras_filtered=residual_bad_cameras,
    )



def process_lightcurve(
    path: str,
    *,
    trigger_mode: str,
    logbf_threshold_dip: float,
    logbf_threshold_jump: float,
    significance_threshold: float,
    p_points: int,
    p_min_dip: float | None,
    p_max_dip: float | None,
    p_min_jump: float | None,
    p_max_jump: float | None,
    mag_points: int,
    mag_min_dip: float | None = None,
    mag_max_dip: float | None = None,
    mag_min_jump: float | None = None,
    mag_max_jump: float | None = None,

    run_min_points: int,
    max_gap_points: int,
    run_max_gap_days: float | None,
    run_min_duration_days: float | None,

    baseline_tag: str,
    baseline_kwargs: dict | None = None,

    compute_event_prob: bool,
    path_metadata: dict | None = None,
    excluded_cameras: str | None = None,
    auto_filter_bad_cameras: bool = False,
    bad_camera_scatter_ratio: float = 2.5,
):
    path = str(path)
    path_metadata = dict(path_metadata or {})

    if os.path.isfile(path) and path.endswith('.csv'):
        df = read_skypatrol_csv(path)
    elif os.path.isfile(path):
        # Handle any dat file extension (.dat, .dat2, .dat3, etc.)
        dir_path = os.path.dirname(path) or '.'
        basename = os.path.basename(path)
        asassn_id = os.path.splitext(basename)[0]
        ext = os.path.splitext(basename)[1][1:] if '.' in basename else None
        dfg, dfv = read_lc_dat2(asassn_id, dir_path, excluded_cameras=excluded_cameras, file_ext=ext)
        df = pd.concat([dfg, dfv], ignore_index=True) if not (dfg.empty and dfv.empty) else pd.DataFrame()
    else:
        raise ValueError(f"Cannot read light curve from path: {path}")

    field_summary = compute_field_summary(df)

    valid_mask = (
        np.isfinite(df["JD"]) &
        np.isfinite(df["mag"]) &
        np.isfinite(df["error"]) &
        (df["error"] > 0) &
        (df["error"] < 10)
    )
    df = df[valid_mask].copy()

    # Only catastrophic unsupported excursions are removed before baseline
    # fitting. Scatter/offset camera decisions happen below in residual space.
    pre_baseline_bad_cameras = set()
    if auto_filter_bad_cameras and "camera#" in df.columns:
        df, pre_baseline_bad_cameras = filter_bad_cameras(
            df,
            lc_path=path,
            filter_scatter=False,
            filter_offset=False,
            filter_catastrophic=True,
            scatter_ratio_threshold=bad_camera_scatter_ratio,
        )

    baseline_func_map = {
        "gp": per_camera_gp_baseline,
        "gp_masked": per_camera_gp_baseline_masked,
        "global_median": global_median_baseline,
        "per_camera_median": per_camera_median_baseline,
        "phase_template": phase_template_baseline,
    }
    baseline_func = baseline_func_map.get(baseline_tag, per_camera_gp_baseline)

    baseline_kwargs_local = dict(DEFAULT_BASELINE_KWARGS if baseline_kwargs is None else baseline_kwargs)
    if baseline_tag == "phase_template":
        period_value = path_metadata.get("pre_periodicity_selected_period")
        try:
            baseline_kwargs_local["period_days"] = float(period_value)
        except (TypeError, ValueError):
            baseline_kwargs_local["period_days"] = np.nan

    # Build mag grids from min/max/points if bounds are provided
    mag_grid_dip = None
    mag_grid_jump = None
    if mag_min_dip is not None and mag_max_dip is not None:
        mag_grid_dip = np.linspace(mag_min_dip, mag_max_dip, mag_points)
    if mag_min_jump is not None and mag_max_jump is not None:
        mag_grid_jump = np.linspace(mag_min_jump, mag_max_jump, mag_points)

    res = score_lightcurve(
        df,
        trigger_mode=trigger_mode,
        logbf_threshold_dip=logbf_threshold_dip,
        logbf_threshold_jump=logbf_threshold_jump,
        significance_threshold=significance_threshold,
        p_points=p_points,
        p_min_dip=p_min_dip,
        p_max_dip=p_max_dip,
        p_min_jump=p_min_jump,
        p_max_jump=p_max_jump,
        mag_points=mag_points,
        mag_grid_dip=mag_grid_dip,
        mag_grid_jump=mag_grid_jump,

        run_min_points=run_min_points,
        max_gap_points=max_gap_points,
        run_max_gap_days=run_max_gap_days,
        run_min_duration_days=run_min_duration_days,

        compute_event_prob=compute_event_prob,
        baseline_func=baseline_func,
        baseline_kwargs=baseline_kwargs_local,
        filter_residual_bad_cameras_enabled=auto_filter_bad_cameras,
        bad_camera_scatter_ratio=bad_camera_scatter_ratio,
    )

    residual_bad_cameras = set(res.get("bad_cameras_filtered", set()) or set())
    bad_cameras_filtered = set(pre_baseline_bad_cameras) | residual_bad_cameras
    df = res.get("df", df)
    n_points = len(df)

    dip = res["dip"]
    jump = res["jump"]

    jd_arr = np.asarray(df["JD"], float)
    jd_first = float(np.nanmin(jd_arr)) if jd_arr.size else np.nan
    jd_last = float(np.nanmax(jd_arr)) if jd_arr.size else np.nan
    cadence_median_days = float(median_dt(jd_arr))

    def max_event_prob(ev):
        ep = ev.get("event_probability")
        if ep is None or (isinstance(ep, float) and not np.isfinite(ep)):
            return np.nan
        ep = np.asarray(ep, float)
        return float(np.nanmax(ep)) if ep.size else np.nan

    def get_best_morph_info(run_list):
        """Extract morphology info, full params, and symmetry from the best run.

        Returns
        -------
        dict with keys: morph, delta_bic, width_param, symmetry,
                        amp, t0, alpha, tau.
        """
        empty = dict(
            morph="none", delta_bic=0.0, width_param=np.nan, symmetry=np.nan,
            amp=np.nan, t0=np.nan, alpha=np.nan, tau=np.nan,
        )
        if not run_list:
            return empty
        best_run = max(run_list, key=lambda x: x['run_max'])

        morph = best_run.get('morphology', 'none')
        delta_bic = best_run.get('delta_bic_null', 0.0)
        symmetry = best_run.get('symmetry_score', np.nan)

        params = best_run.get('params', {})

        # Main width parameter (backward-compatible)
        if morph == 'gaussian':
            width_param = params.get('sigma', np.nan)
        elif morph == 'skew_gaussian':
            width_param = params.get('sigma', np.nan)
        elif morph == 'paczynski':
            width_param = params.get('tE', np.nan)
        elif morph == 'fred':
            width_param = params.get('tau', np.nan)
        else:
            width_param = np.nan

        amp = params.get('amp', np.nan)
        t0 = params.get('t0', np.nan)
        alpha = params.get('alpha', np.nan)      # skew_gaussian only
        tau = params.get('tau', np.nan)           # fred only

        return dict(
            morph=str(morph),
            delta_bic=float(delta_bic),
            width_param=float(width_param) if np.isfinite(width_param) else np.nan,
            symmetry=float(symmetry),
            amp=float(amp) if np.isfinite(amp) else np.nan,
            t0=float(t0) if np.isfinite(t0) else np.nan,
            alpha=float(alpha) if np.isfinite(alpha) else np.nan,
            tau=float(tau) if np.isfinite(tau) else np.nan,
        )

    dip_mi = get_best_morph_info(dip["run_summaries"])
    jump_mi = get_best_morph_info(jump["run_summaries"])

    dip_recurrence = compute_recurrence_stats(dip["run_summaries"])
    jump_recurrence = compute_recurrence_stats(jump["run_summaries"])

    cams = df["camera#"].dropna() if "camera#" in df.columns else pd.Series([], dtype=str)

    unique_cams = np.unique(cams.astype(str)) if len(cams) > 0 else np.array([], dtype=str)
    n_cameras = int(unique_cams.size)
    cam_counts = cams.value_counts() if len(cams) > 0 else pd.Series([], dtype=int)
    camera_min_points = int(cam_counts.min()) if len(cam_counts) else 0
    camera_max_points = int(cam_counts.max()) if len(cam_counts) else 0
    camera_ids = ",".join(unique_cams) if len(unique_cams) > 0 else ""

    dipper_score = 0.0
    dipper_n_dips = 0
    dipper_n_valid_dips = 0
    if bool(dip["significant"]):
        # Use computed baseline for scoring
        df_base = res.get("df_base")
        if df_base is not None and "baseline" in df_base.columns:
            baseline_mags = df_base["baseline"].to_numpy()
        else:
            baseline_mags = None
        score, events = compute_event_score(df, event_type='dip', baseline_mags=baseline_mags)
        dipper_score = float(score)

        dipper_n_dips = int(len(events))
        dipper_n_valid_dips = int(sum(1 for e in events if e.valid))

    jumper_score = 0.0
    jumper_n_jumps = 0
    jumper_n_valid_jumps = 0
    if bool(jump["significant"]):
        df_base = res.get("df_base")
        if df_base is not None and "baseline" in df_base.columns:
            baseline_mags = df_base["baseline"].to_numpy()
        else:
            baseline_mags = None
        score, events = compute_event_score(df, event_type='jump', baseline_mags=baseline_mags)
        jumper_score = float(score)

        jumper_n_jumps = int(len(events))
        jumper_n_valid_jumps = int(sum(1 for e in events if e.valid))

    lc_path = str(path)
    asas_sn_id = Path(path).stem

    return dict(
        candidate_id=f"stv_{asas_sn_id}",
        timescale="stv",
        asas_sn_id=asas_sn_id,
        lc_path=lc_path,

        dip_significant=bool(dip["significant"]),
        jump_significant=bool(jump["significant"]),

        n_points=int(n_points),
        jd_first=jd_first,
        jd_last=jd_last,
        cadence_median_days=cadence_median_days,

        dip_best_morph=str(dip_mi["morph"]),
        dip_best_delta_bic=float(dip_mi["delta_bic"]),
        dip_best_width_param=float(dip_mi["width_param"]),
        dip_symmetry_score=float(dip_mi["symmetry"]),
        dip_best_amp=float(dip_mi["amp"]),
        dip_best_t0=float(dip_mi["t0"]),
        dip_best_alpha=float(dip_mi["alpha"]),
        dip_best_tau=float(dip_mi["tau"]),
        jump_best_morph=str(jump_mi["morph"]),
        jump_best_delta_bic=float(jump_mi["delta_bic"]),
        jump_best_width_param=float(jump_mi["width_param"]),
        jump_best_amp=float(jump_mi["amp"]),
        jump_best_t0=float(jump_mi["t0"]),
        jump_best_alpha=float(jump_mi["alpha"]),
        jump_best_tau=float(jump_mi["tau"]),
        dip_count=int(len(dip["event_indices"])),
        jump_count=int(len(jump["event_indices"])),

        dip_run_count=int(dip.get("n_runs", 0)),
        jump_run_count=int(jump.get("n_runs", 0)),

        dip_max_run_points=int(dip.get("max_run_points", 0)),
        jump_max_run_points=int(jump.get("max_run_points", 0)),
        dip_max_run_duration=float(dip.get("max_run_duration", np.nan)),
        jump_max_run_duration=float(jump.get("max_run_duration", np.nan)),
        dip_max_run_sum=float(dip.get("max_run_sum", np.nan)),
        jump_max_run_sum=float(jump.get("max_run_sum", np.nan)),
        dip_max_run_max=float(dip.get("max_run_max", np.nan)),
        jump_max_run_max=float(jump.get("max_run_max", np.nan)),
        dip_max_run_cameras=int(dip.get("max_run_cameras", 0)),
        jump_max_run_cameras=int(jump.get("max_run_cameras", 0)),

        dip_max_log_bf_local=float(dip.get("max_log_bf_local", np.nan)),
        jump_max_log_bf_local=float(jump.get("max_log_bf_local", np.nan)),

        dip_bayes_factor=float(dip["bayes_factor"]),
        jump_bayes_factor=float(jump["bayes_factor"]),

        baseline_mag=float(dip.get("baseline_mag", jump.get("baseline_mag", np.nan))),
        dip_best_p=float(dip["best_p"]),
        jump_best_p=float(jump["best_p"]),
        dip_best_mag_event=float(dip.get("best_mag_event", np.nan)),
        jump_best_mag_event=float(jump.get("best_mag_event", np.nan)),
        dip_trigger_max=float(dip.get("trigger_max", np.nan)),
        jump_trigger_max=float(jump.get("trigger_max", np.nan)),
        dip_max_event_prob=max_event_prob(dip),
        jump_max_event_prob=max_event_prob(jump),

        n_cameras=int(n_cameras),
        camera_ids=str(camera_ids),
        camera_min_points=int(camera_min_points),
        camera_max_points=int(camera_max_points),
        asassn_field_key=str(field_summary.get("asassn_field_key", "")),
        asassn_fields=str(field_summary.get("asassn_fields", "")),
        asassn_field_count=int(field_summary.get("asassn_field_count") or 0),
        asassn_field_key_fraction=float(field_summary.get("asassn_field_key_fraction", np.nan)),
        camera_field_key=str(field_summary.get("camera_field_key", "")),
        camera_fields=str(field_summary.get("camera_fields", "")),
        camera_field_count=int(field_summary.get("camera_field_count") or 0),
        camera_field_key_fraction=float(field_summary.get("camera_field_key_fraction", np.nan)),

        dipper_score=float(dipper_score),
        dipper_n_dips=int(dipper_n_dips),
        dipper_n_valid_dips=int(dipper_n_valid_dips),

        jumper_score=float(jumper_score),
        jumper_n_jumps=int(jumper_n_jumps),
        jumper_n_valid_jumps=int(jumper_n_valid_jumps),

        baseline_source=str(dip.get("baseline_source", jump.get("baseline_source", "unknown"))),
        trigger_mode=str(trigger_mode),
        dip_trigger_threshold=float(dip.get("trigger_threshold", np.nan)),
        jump_trigger_threshold=float(jump.get("trigger_threshold", np.nan)),
        bad_cameras_filtered=",".join(str(c) for c in sorted(bad_cameras_filtered, key=str)) if bad_cameras_filtered else "",

        # Recurrence statistics
        dip_is_single_event=bool(dip_recurrence["is_single_event"]),
        dip_inter_event_spacing_median=float(dip_recurrence["inter_event_spacing_median"]),
        dip_inter_event_spacing_std=float(dip_recurrence["inter_event_spacing_std"]),
        dip_amplitude_consistency=float(dip_recurrence["amplitude_consistency"]),
        dip_duration_consistency=float(dip_recurrence["duration_consistency"]),

        jump_is_single_event=bool(jump_recurrence["is_single_event"]),
        jump_inter_event_spacing_median=float(jump_recurrence["inter_event_spacing_median"]),
        jump_inter_event_spacing_std=float(jump_recurrence["inter_event_spacing_std"]),
        jump_amplitude_consistency=float(jump_recurrence["amplitude_consistency"]),
        jump_duration_consistency=float(jump_recurrence["duration_consistency"]),
    )


# Shared config for ProcessPoolExecutor workers (set once per worker via initializer)
_worker_config: dict = {}


def _init_worker(config: dict) -> None:
    """Store config in this worker process so _process_one can read it (avoids re-serializing for every path)."""
    global _worker_config
    _worker_config.clear()
    _worker_config.update(config)


def _process_one(path: str, path_metadata: dict | None) -> dict:
    """Process a single light curve using config from _worker_config. Called by executor with minimal per-task args."""
    return process_lightcurve(path, path_metadata=path_metadata, **_worker_config)


def _iter_batches(items: list, batch_size: int):
    batch_size = max(1, int(batch_size))
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def _process_batch(batch: list[tuple[str, dict | None]]) -> list[dict]:
    """Process a small batch in one worker while preserving per-path failures."""
    batch_results: list[dict] = []
    for path, path_metadata in batch:
        try:
            result = _process_one(path, path_metadata)
            batch_results.append(
                {
                    "lc_path": str(path),
                    "result": result,
                    "error": None,
                    "traceback": None,
                }
            )
        except Exception as e:
            batch_results.append(
                {
                    "lc_path": str(path),
                    "result": None,
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                }
            )
    return batch_results


def _resolve_events_chunk_size(value: int | None) -> int | None:
    if value is None:
        return EVENTS_OUTPUT_CHUNK_SIZE
    if value > 0:
        return int(value)
    return None


def _event_task_batch_size(task_chunk_size: int, n_workers: int) -> int:
    target_per_worker = max(1, task_chunk_size // max(1, int(n_workers) * 8))
    return max(1, min(32, target_per_worker, task_chunk_size))


def main():
    parser = argparse.ArgumentParser(description="Run Bayesian event scoring on light curves in parallel.")
    g_input = parser.add_argument_group("Input")
    g_output = parser.add_argument_group("Output")
    g_detection = parser.add_argument_group("Detection")
    g_runs = parser.add_argument_group("Run grouping")
    g_baseline = parser.add_argument_group("Baseline")
    g_cleaning = parser.add_argument_group("Cleaning")
    g_general = parser.add_argument_group("General")
    g_input.add_argument("--input", dest="input_patterns", action="append", default=None, help="Path or glob to a light-curve file. Repeat for multiple inputs.")
    g_input.add_argument("--input-file", type=Path, default=None, help="Read paths from file (one path per line); avoids long argv for large batches.")
    g_input.add_argument("--mag-bin", dest="mag_bins", action="append", choices=MAG_BINS, help="Process all light curves in this magnitude bin.")
    g_input.add_argument("--lc-path", type=str, default=str(LCV2_ROOT), help="Base path to light curve directories")
    g_input.add_argument("--workers", type=int, default=WORKERS, help="Number of worker processes")

    g_output.add_argument("--output", type=Path, default=None, help="Output path for results (suffix adjusted per format).")
    g_output.add_argument("--output-format", choices=("parquet", "parquet_chunk"), default=OUTPUT_FORMAT, help="Output format for results.")
    g_output.add_argument("--error-output", type=Path, default=None, help="Optional Parquet path for per-light-curve processing errors.")
    g_output.add_argument("--metadata", type=Path, default=None, help="Optional Parquet with 'lc_path' and extra metadata columns to attach to results.")
    g_output.add_argument("--chunk-size", type=int, help="Number of result rows per output chunk; <=0 buffers until the end.")

    g_detection.add_argument("--trigger-mode", choices=("posterior_prob", "logbf"), help="Triggering criterion to use.")
    g_detection.add_argument("--logbf-threshold-dip", type=float, help="Minimum local log Bayes factor for dip triggers.")
    g_detection.add_argument("--logbf-threshold-jump", type=float, help="Minimum local log Bayes factor for jump triggers.")
    g_detection.add_argument("--significance-threshold", type=float, help="Posterior probability significance threshold.")
    g_detection.add_argument("--p-points", type=int, help="Number of posterior-probability grid points.")
    g_detection.add_argument("--p-min-dip", type=float, help="Minimum dip posterior-probability grid value.")
    g_detection.add_argument("--p-max-dip", type=float, help="Maximum dip posterior-probability grid value.")
    g_detection.add_argument("--p-min-jump", type=float, help="Minimum jump posterior-probability grid value.")
    g_detection.add_argument("--p-max-jump", type=float, help="Maximum jump posterior-probability grid value.")
    g_detection.add_argument("--mag-points", type=int, help="Number of magnitude-offset grid points.")
    g_detection.add_argument("--mag-min-dip", type=float, help="Minimum dip magnitude-offset grid value.")
    g_detection.add_argument("--mag-max-dip", type=float, help="Maximum dip magnitude-offset grid value.")
    g_detection.add_argument("--mag-min-jump", type=float, help="Minimum jump magnitude-offset grid value.")
    g_detection.add_argument("--mag-max-jump", type=float, help="Maximum jump magnitude-offset grid value.")
    g_detection.add_argument("--min-mag-offset", type=float, help="Minimum absolute magnitude offset considered for events.")
    g_detection.add_argument("--no-event-prob", action="store_true", help="Skip event-probability columns; incompatible with posterior_prob triggering.")

    g_runs.add_argument("--run-min-points", type=int, help="Minimum points required in an event run.")
    g_runs.add_argument("--run-max-gap-points", type=int, help="Maximum missing/non-trigger points allowed inside a run.")
    g_runs.add_argument("--run-max-gap-days", type=float, help="Maximum time gap allowed inside a run.")
    g_runs.add_argument("--run-min-duration-days", type=float, help="Minimum event-run duration in days.")

    g_baseline.add_argument(
        "--baseline-func",
        choices=("gp", "gp_masked", "global_median", "per_camera_median", "phase_template"),
        help="Baseline model used before event scoring.",
    )
    g_baseline.add_argument("--baseline-s0", type=float, help="SHOTerm S0 for GP baselines.")
    g_baseline.add_argument("--baseline-w0", type=float, help="SHOTerm w0 for GP baselines.")
    g_baseline.add_argument("--baseline-q", type=float, help="SHOTerm Q for GP baselines.")
    g_baseline.add_argument("--baseline-jitter", type=float, help="Additional jitter for GP baselines.")
    g_baseline.add_argument("--baseline-sigma-floor", type=float, help="Optional fixed sigma floor for GP baselines.")

    g_cleaning.add_argument("--filter-bad-cameras", dest="filter_bad_cameras", action="store_true", help="Enable automatic bad-camera filtering.")
    g_cleaning.add_argument("--no-filter-bad-cameras", dest="filter_bad_cameras", action="store_false", help="Disable automatic bad-camera filtering.")
    g_cleaning.add_argument("--bad-camera-scatter-ratio", type=float, help="Scatter ratio threshold for automatic bad-camera filtering.")
    g_cleaning.add_argument("--max-error-fraction", type=float, help="Abort when processing failures exceed this fraction.")

    add_config_args(g_general)
    g_general.add_argument("-o", "--overwrite", action="store_true", help="Overwrite checkpoint log and existing output if present (start fresh).")
    g_general.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output (default: quiet).")
    g_general.add_argument("--quiet", action="store_true", help=argparse.SUPPRESS)
    parser.set_defaults(**EVENTS_CONFIG_DEFAULTS)

    args = parser.parse_args()
    apply_config(
        args,
        command="events",
        valid_keys=namespace_keys(parser, EVENTS_CONFIG_DEFAULTS),
        path_keys=EVENTS_CONFIG_PATH_KEYS,
    )
    if args.trigger_mode == "posterior_prob" and args.no_event_prob:
        raise SystemExit("posterior_prob triggering requires event_prob; remove --no-event-prob")
    if not (0.0 <= args.max_error_fraction <= 1.0):
        raise SystemExit("--max-error-fraction must be between 0 and 1")

    compute_event_prob = (not args.no_event_prob)
    baseline_tag = args.baseline_func

    output_format = args.output_format.lower()
    quiet = not args.verbose

    def default_output_dir() -> Path:
        base_dir = DEFAULT_OUTPUT_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return base_dir / "runs" / timestamp / "results"

    if not args.output:
        out_dir = default_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(out_dir / "lc_events_results.parquet")

    meta_df: pd.DataFrame | None = None
    metadata_by_path = None
    if args.metadata:
        meta_df = pd.read_parquet(args.metadata)
        if "lc_path" not in meta_df.columns:
            raise SystemExit("metadata parquet must include an 'lc_path' column")
        meta_df["lc_path"] = meta_df["lc_path"].astype(str)
        metadata_by_path = meta_df.set_index("lc_path").to_dict(orient="index")

    events_writer_columns = build_events_writer_columns(meta_df.columns if meta_df is not None else None)
    events_column_kinds = infer_events_metadata_column_kinds(meta_df)

    def ensure_suffix(path: Path | None, fmt: str) -> Path | None:
        if path is None:
            return None
        suffix_map = {"parquet": ".parquet", "parquet_chunk": None}
        ext = suffix_map.get(fmt)
        if ext and path.suffix.lower() != ext:
            return path.with_suffix(ext)
        return path

    def collect_processed_from_output(path: Path | None, fmt: str) -> set[str]:
        if path is None or (not path.exists()):
            return set()
        try:
            if fmt == "parquet":
                table = pq.read_table(path, columns=["lc_path"])
                df_existing = table.to_pandas()
            elif fmt == "parquet_chunk":

                dataset = ds.dataset(path, format="parquet")
                table = dataset.to_table(columns=["lc_path"])
                df_existing = table.to_pandas()
            else:
                return set()
            if "lc_path" in df_existing.columns:
                return set(df_existing["lc_path"].astype(str))
        except Exception as e:
            _log(f"Warning: could not read existing output {path} to skip duplicates: {e}", quiet)
        return set()

    def clear_existing_output(path: Path | None, fmt: str) -> None:
        if path is None or (not path.exists()):
            return
        try:
            if fmt == "parquet_chunk" and path.is_dir():
                removed_any = False
                for child in path.glob("chunk_*.parquet*"):
                    child.unlink()
                    removed_any = True
                if removed_any:
                    _log(f"Overwriting existing output chunks in {path}", quiet)
            else:
                path.unlink()
                _log(f"Overwriting existing output file: {path}", quiet)
        except Exception as e:
            _log(f"Warning: Could not remove existing output {path} ({e}). Will append.", quiet)

    # checkpoint
    base_output_path = ensure_suffix(Path(args.output).expanduser() if args.output else None, output_format)
    if args.mag_bins and base_output_path is not None:
        # pick the bin name if only one was given; otherwise use the "multi" tag
        bin_tag = args.mag_bins[0] if len(args.mag_bins) == 1 else "multi"
        base_output_path = base_output_path.with_name(f"{base_output_path.stem}_{bin_tag}{base_output_path.suffix}")

    if base_output_path:
        checkpoint_log = base_output_path.with_name(f"{base_output_path.stem}_PROCESSED.txt")
    else:
        checkpoint_log = None

    def default_error_output_path(path: Path | None) -> Path | None:
        if path is None:
            return None
        if output_format == "parquet_chunk":
            return path.parent / f"{path.name}_ERRORS.parquet"
        return path.with_name(f"{path.stem}_ERRORS.parquet")

    error_output_path = (
        Path(args.error_output).expanduser()
        if args.error_output
        else default_error_output_path(base_output_path)
    )

    processed_files = set()
    if checkpoint_log and checkpoint_log.exists() and args.overwrite:
        try:
            with open(checkpoint_log, "w"):
                pass
            _log(f"Overwriting checkpoint log: {checkpoint_log}", quiet)
        except Exception as e:
            _log(f"Warning: Could not overwrite checkpoint file ({e}). Continuing without resume.", quiet)

    if args.overwrite:
        clear_existing_output(base_output_path, output_format)
        if error_output_path and error_output_path.exists():
            try:
                error_output_path.unlink()
            except Exception as e:
                _log(f"Warning: Could not remove existing error output {error_output_path} ({e}). Will append.", quiet)

    if checkpoint_log and checkpoint_log.exists() and not args.overwrite:
        _log("--- RESUME DETECTED ---", quiet)
        _log(f"Reading processed files from: {checkpoint_log}", quiet)
        try:
            with open(checkpoint_log, "r") as f:
                processed_files = set(line.strip() for line in f)
            _log(f"Found {len(processed_files)} previously processed files.", quiet)
        except Exception as e:
            _log(f"Warning: Could not read checkpoint file ({e}). Starting fresh.", quiet)

    # existing output (avoid duplicates if checkpoint was out-of-sync)
    if not args.overwrite:
        processed_files |= collect_processed_from_output(base_output_path, output_format)

    input_patterns: list[str] = []
    if args.input_patterns:
        input_patterns.extend(args.input_patterns)
    if args.input_file:
        with open(args.input_file) as f:
            for line in f:
                p = line.strip()
                if p:
                    input_patterns.append(p)

    expanded_inputs = []
    if args.mag_bins:
        lc_path = args.lc_path
        for mag_bin in args.mag_bins:
            mag_bin_dir = os.path.join(lc_path, mag_bin)
            lc_dirs = sorted(glob.glob(os.path.join(mag_bin_dir, "lc*_cal")))
            for lc_dir in lc_dirs:
                csv_files = sorted(glob.glob(os.path.join(lc_dir, "*.csv")))
                dat_files = sorted(glob.glob(os.path.join(lc_dir, "*.dat*")))
                if csv_files: expanded_inputs.extend(csv_files)
                elif dat_files: expanded_inputs.extend(dat_files)
    
    for pattern in input_patterns:
        if '*' in pattern or '?' in pattern or '[' in pattern:
            matches = glob.glob(pattern)
            if matches:
                expanded_inputs.extend(sorted(matches))
            else:
                _log(f"Warning: glob pattern '{pattern}' matched no files", quiet)
        else: expanded_inputs.append(pattern)
    
    seen = set()
    expanded_inputs = [x for x in expanded_inputs if not (x in seen or seen.add(x))]
    
    if not expanded_inputs: raise SystemExit("No input files found.")
    
    # --- CHECKPOINT FILTERING ---
    original_count = len(expanded_inputs)
    expanded_inputs = [x for x in expanded_inputs if str(x) not in processed_files]
    _log(f"Processing {len(expanded_inputs)} light curve file(s) (Filtered from {original_count})...", quiet)
    
    if len(expanded_inputs) == 0:
        _log("All files have been processed according to checkpoint! Exiting.", quiet)
        return

    attempted_count = len(expanded_inputs)

    results = []
    errors = []
    
    chunk_size = _resolve_events_chunk_size(args.chunk_size)
    task_chunk_size = max(1, int(chunk_size or EVENTS_OUTPUT_CHUNK_SIZE))
    task_batch_size = _event_task_batch_size(task_chunk_size, args.workers)
    event_batch_total = (len(expanded_inputs) + task_chunk_size - 1) // task_chunk_size

    total_written = 0
    total_dip_sig = 0
    total_jump_sig = 0
    total_any_sig = 0

    class ParquetChunkWriter:
        def __init__(
            self,
            path: Path,
            schema_columns: Iterable[str],
            column_kinds: Mapping[str, str] | None = None,
        ):
            self.path = Path(path)
            self.schema_columns = list(schema_columns)
            self.column_kinds = dict(column_kinds or {})
            self.append = self.path.exists() and self.path.stat().st_size > 0

        def write_chunk(self, chunk_results):
            if not chunk_results:
                return
            df_chunk = normalize_events_frame(
                pd.DataFrame(chunk_results),
                self.schema_columns,
                column_kinds=self.column_kinds,
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.append:
                try:
                    existing_df = pq.read_table(self.path).to_pandas()
                except Exception as e:
                    _log(f"Warning: could not read existing {self.path}, starting fresh: {e}", False)
                    existing_df = None
                if existing_df is not None:
                    existing_df = normalize_events_frame(
                        existing_df,
                        self.schema_columns,
                        column_kinds=self.column_kinds,
                    )
                    df_chunk = pd.concat([existing_df, df_chunk], ignore_index=True, sort=False)
            table = events_table_from_frame(
                df_chunk,
                self.schema_columns,
                column_kinds=self.column_kinds,
            )
            tmp_path = self.path.with_suffix('.parquet.tmp')
            pq.write_table(table, tmp_path, compression=PARQUET_OUTPUT_COMPRESSION)
            os.replace(tmp_path, self.path)
            self.append = True

        def close(self):
            return

    class ParquetDatasetWriter:
        def __init__(
            self,
            path: Path,
            schema_columns: Iterable[str],
            column_kinds: Mapping[str, str] | None = None,
        ):
            self.path = Path(path)
            self.schema_columns = list(schema_columns)
            self.column_kinds = dict(column_kinds or {})
            self.path.mkdir(parents=True, exist_ok=True)
            existing = sorted(self.path.glob("chunk_*.parquet"))
            if existing:
                try:
                    last = existing[-1].stem.split("_")[-1]
                    self.counter = int(last) + 1
                except Exception:
                    self.counter = len(existing)
            else:
                self.counter = 0

        def write_chunk(self, chunk_results):
            if not chunk_results:
                return
            table = events_table_from_frame(
                pd.DataFrame(chunk_results),
                self.schema_columns,
                column_kinds=self.column_kinds,
            )
            tmp_path = self.path / f"chunk_{self.counter:06d}.parquet.tmp"
            final_path = self.path / f"chunk_{self.counter:06d}.parquet"
            pq.write_table(table, tmp_path, compression=PARQUET_OUTPUT_COMPRESSION)
            os.replace(tmp_path, final_path)
            self.counter += 1

        def close(self):
            return

    def make_writer(path: Path | None, fmt: str):
        if path is None:
            return None
        if fmt == "parquet":
            return ParquetChunkWriter(path, events_writer_columns, events_column_kinds)
        elif fmt == "parquet_chunk":
            return ParquetDatasetWriter(path, events_writer_columns, events_column_kinds)
        else:
            raise ValueError(f"Unknown output format: {fmt}")

    output_path = base_output_path
    writer = make_writer(output_path, output_format)
    if output_path:
        args.output = str(output_path)

    def count_significant(rows: list[dict]) -> tuple[int, int, int]:
        dip = 0
        jump = 0
        any_sig = 0
        for row in rows:
            dip_sig = bool(row.get("dip_significant"))
            jump_sig = bool(row.get("jump_significant"))
            if dip_sig:
                dip += 1
            if jump_sig:
                jump += 1
            if dip_sig or jump_sig:
                any_sig += 1
        return dip, jump, any_sig

    def write_errors(error_rows: list[dict]) -> None:
        if not error_rows or error_output_path is None:
            return
        try:
            df_errors = pd.DataFrame(error_rows)
            error_output_path.parent.mkdir(parents=True, exist_ok=True)
            append = error_output_path.exists() and (not args.overwrite)
            if append:
                df_existing = pd.read_parquet(error_output_path)
                df_errors = pd.concat([df_existing, df_errors], ignore_index=True, sort=False)
            df_errors.to_parquet(error_output_path, index=False, compression=PARQUET_OUTPUT_COMPRESSION)
            _log(f"Wrote {len(df_errors)} processing errors to {error_output_path}", quiet)
        except Exception as e:
            print(f"Warning: could not write processing errors to {error_output_path}: {e}", flush=True)

    def write_chunk(chunk_results, is_final=False):
        if not chunk_results: 
            return
        nonlocal total_written, total_dip_sig, total_jump_sig, total_any_sig, writer
        
        # Tag signal amplitude failures if requested
        if args.min_mag_offset is not None and args.min_mag_offset > 0:
            df_chunk = pd.DataFrame(chunk_results)
            dip_diff = np.abs(df_chunk["dip_best_mag_event"] - df_chunk["baseline_mag"])
            jump_diff = np.abs(df_chunk["jump_best_mag_event"] - df_chunk["baseline_mag"])
            passed = (dip_diff > args.min_mag_offset) | (jump_diff > args.min_mag_offset)
            df_chunk["failed_signal_amplitude"] = ~passed
            n_failed = int((~passed).sum())
            if n_failed > 0:
                _log(f"Signal amplitude filter: {n_failed}/{len(df_chunk)} failed", quiet)
            chunk_results = df_chunk.to_dict('records')
        
        if writer is not None:
            writer.write_chunk(chunk_results)

        if checkpoint_log:
            checkpoint_log.parent.mkdir(parents=True, exist_ok=True)
            mode = "a"
            with open(checkpoint_log, mode) as f:
                for row in chunk_results:
                    f.write(str(row["lc_path"]) + "\n")

        chunk_dip, chunk_jump, chunk_any = count_significant(chunk_results)
        total_dip_sig += chunk_dip
        total_jump_sig += chunk_jump
        total_any_sig += chunk_any
        total_written += len(chunk_results)
        if is_final:
            if writer is not None:
                writer.close()
            if args.output:
                _log(
                    f"Wrote {total_written} total rows to {args.output} "
                    f"(dip_sig={total_dip_sig}, jump_sig={total_jump_sig}, any_sig={total_any_sig})"
                , quiet)
        else:
            _log(
                f"Wrote chunk: {len(chunk_results)} rows (total: {total_written}) "
                f"(dip_sig={total_dip_sig}, jump_sig={total_jump_sig}, any_sig={total_any_sig})"
            , quiet)

    # Build shared config once (serialized once per worker via initializer, not per path)
    baseline_kwargs = dict(
        S0=args.baseline_s0,
        w0=args.baseline_w0,
        q=args.baseline_q,
        jitter=args.baseline_jitter,
        sigma_floor=args.baseline_sigma_floor,
        add_sigma_eff_col=True,
    )
    worker_config = {
        "trigger_mode": args.trigger_mode,
        "logbf_threshold_dip": args.logbf_threshold_dip,
        "logbf_threshold_jump": args.logbf_threshold_jump,
        "significance_threshold": args.significance_threshold,
        "p_points": args.p_points,
        "p_min_dip": args.p_min_dip,
        "p_max_dip": args.p_max_dip,
        "p_min_jump": args.p_min_jump,
        "p_max_jump": args.p_max_jump,
        "mag_points": args.mag_points,
        "mag_min_dip": args.mag_min_dip,
        "mag_max_dip": args.mag_max_dip,
        "mag_min_jump": args.mag_min_jump,
        "mag_max_jump": args.mag_max_jump,
        "run_min_points": args.run_min_points,
        "max_gap_points": args.run_max_gap_points,
        "run_max_gap_days": args.run_max_gap_days,
        "run_min_duration_days": args.run_min_duration_days,
        "baseline_tag": baseline_tag,
        "baseline_kwargs": baseline_kwargs,
        "compute_event_prob": compute_event_prob,
        "auto_filter_bad_cameras": args.filter_bad_cameras,
        "bad_camera_scatter_ratio": args.bad_camera_scatter_ratio,
    }

    def make_task(path: str) -> tuple[str, dict | None]:
        path_meta = None
        if metadata_by_path:
            meta = metadata_by_path.get(str(path))
            if meta:
                path_meta = dict(meta)
        return str(path), path_meta

    def handle_batch_result(batch_result: dict) -> None:
        nonlocal results
        path = str(batch_result["lc_path"])
        error = batch_result.get("error")
        if error is not None:
            tb_str = str(batch_result.get("traceback") or "")
            errors.append(dict(lc_path=path, error=error, traceback=tb_str))
            print(f"ERROR processing {path}: {error}", flush=True)
            if "too many values to unpack" in error:
                print(f"Full traceback:\n{tb_str}", flush=True)
            return

        result = dict(batch_result["result"])
        if metadata_by_path:
            meta = metadata_by_path.get(path)
            if meta:
                result.update(meta)
        results.append(result)
        if chunk_size and len(results) >= chunk_size:
            write_chunk(results)
            results = []

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(worker_config,),
    ) as ex, tqdm(
        total=event_batch_total,
        desc="Event batches",
        unit="batch",
        position=0,
        disable=quiet,
    ) as batch_pbar, tqdm(
        total=len(expanded_inputs),
        desc="LCs",
        unit="lc",
        position=1,
        disable=quiet,
    ) as pbar:
        for chunk_start in range(0, len(expanded_inputs), task_chunk_size):
            path_chunk = expanded_inputs[chunk_start:chunk_start + task_chunk_size]
            batch_pbar.set_postfix(rows=len(path_chunk), refresh=False)
            task_batches = list(_iter_batches([make_task(path) for path in path_chunk], task_batch_size))
            futs = {ex.submit(_process_batch, batch): batch for batch in task_batches}

            for fut in as_completed(futs):
                batch = futs[fut]
                try:
                    batch_results = fut.result()
                except Exception as e:
                    tb_str = traceback.format_exc()
                    for path, _path_meta in batch:
                        errors.append(dict(lc_path=str(path), error=repr(e), traceback=tb_str))
                        print(f"ERROR processing {path}: {e}", flush=True)
                    pbar.update(len(batch))
                    continue

                for batch_result in batch_results:
                    handle_batch_result(batch_result)
                pbar.update(len(batch_results))
            batch_pbar.update(1)

    if results:
        write_chunk(results, is_final=True)
    elif args.output and total_written == 0:
        pass
    else:
        if not quiet:
            for row in results:
                print(f"{row['lc_path']}\tmode={row['trigger_mode']}\tdip_sig={row['dip_significant']} jump_sig={row['jump_significant']}")

    if errors:
        write_errors(errors)
        error_fraction = len(errors) / attempted_count if attempted_count else 0.0
        print(
            f"Completed with {len(errors)}/{attempted_count} failures "
            f"({error_fraction:.2%}); wrote {total_written} result rows.",
            flush=True,
        )
        if error_fraction > args.max_error_fraction:
            print(
                "Failure fraction exceeded "
                f"--max-error-fraction={args.max_error_fraction:.2%}; aborting.",
                flush=True,
            )
            raise SystemExit(2)


if __name__ == "__main__":
    main()
