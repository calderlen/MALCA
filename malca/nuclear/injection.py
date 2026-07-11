"""
Injection-recovery benchmark for nuclear AGN/TDE/CLAGN arbitration.

This module injects analytic nuclear-transient templates into real ASAS-SN
light curves, attaches synthetic context fields, scores the rows with the
existing nuclear scorer, and records whether arbitration recovers the injected
truth class.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from malca.config import (
    ASASSN_INDEX_PATH,
    NUCLEAR_INJECTION_AMP_MAX,
    NUCLEAR_INJECTION_AMP_MIN,
    NUCLEAR_INJECTION_AMP_STEPS,
    NUCLEAR_INJECTION_CHECKPOINT_INTERVAL,
    NUCLEAR_INJECTION_CHUNK_SIZE,
    NUCLEAR_INJECTION_CLASSES,
    NUCLEAR_INJECTION_CONTROL_SAMPLE_SIZE,
    NUCLEAR_INJECTION_MIN_MARGIN,
    NUCLEAR_INJECTION_MIN_SCORE,
    NUCLEAR_INJECTION_OUTPUT_DIR,
    NUCLEAR_INJECTION_REPEATS_PER_GRID,
    NUCLEAR_INJECTION_TIMESCALE_MAX_DAYS,
    NUCLEAR_INJECTION_TIMESCALE_MIN_DAYS,
    NUCLEAR_INJECTION_TIMESCALE_STEPS,
)
from malca.io.table_io import read_parquet_table, write_parquet_table
from malca.nuclear.arbitration import arbitrate_nuclear_scores
from malca.nuclear.features import compute_nuclear_lightcurve_features
from malca.nuclear.redshift import resolve_redshift_spectral_types
from malca.nuclear.scoring import score_nuclear_candidates
from malca.plotting.lightcurve_publication import (
    FIG_SINGLE_COL_HEATMAP,
    FIG_SINGLE_COL_LC_WIDE,
    apply_publication_rcparams,
    save_publication_figure,
)

apply_publication_rcparams(plt)


DAT2_COLUMNS = [
    "JD",
    "mag",
    "error",
    "good_bad",
    "camera",
    "v_g_band",
    "saturated",
    "cam_field",
]

TRUTH_CLASSES = ("agn", "tde", "clagn", "control")
SCORE_COLUMNS = ("agn_prior_score", "tde_candidate_score", "clagn_photometric_score")
MJD_OFFSET = 2400000.5

_GLOBAL: dict[str, object] = {}


@dataclass(frozen=True)
class TrialSpec:
    trial_index: int
    truth_class: str
    amplitude_mag: float
    timescale_days: float
    repeat_index: int = 0
    source_index: int | None = None


class ParquetAppendWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.columns = None
        if self.path.exists() and self.path.stat().st_size > 0:
            try:
                self.columns = read_parquet_table(self.path).columns.tolist()
            except Exception:
                self.columns = None

    def write_chunk(self, chunk_results: list[dict]) -> None:
        if not chunk_results:
            return
        df_chunk = pd.DataFrame(chunk_results)
        if self.columns is None:
            self.columns = list(df_chunk.columns)
        else:
            new_columns = [col for col in df_chunk.columns if col not in self.columns]
            if new_columns:
                self.columns = [*self.columns, *new_columns]
        df_chunk = df_chunk.reindex(columns=self.columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size > 0:
            existing = read_parquet_table(self.path).reindex(columns=self.columns)
            df_chunk = pd.concat([existing, df_chunk], ignore_index=True)
        write_parquet_table(df_chunk, self.path)

    def close(self) -> None:
        return


def _write_checkpoint(path: Path, last_index: int) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(str(int(last_index)), encoding="ascii")
    tmp_path.replace(path)


def _read_checkpoint(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="ascii").strip()
        if text:
            return int(text)
    except Exception:
        return None
    return None


def _get_id_col(df: pd.DataFrame) -> str:
    for col in ("asas_sn_id", "ASAS-SN ID", "source_id", "candidate_id", "id"):
        if col in df.columns:
            return col
    raise KeyError("Manifest is missing a usable ID column.")


def _normalize_id_values(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().fillna("").astype(str)


def _truthy_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return series.astype("string").str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _parquet_columns(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq

        return set(pq.read_schema(path).names)
    except Exception:
        return set(read_parquet_table(path).columns)


def _filter_existing_dat_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "dat_exists" not in df.columns:
        return df
    out = df.loc[_truthy_series(df["dat_exists"])].copy()
    if out.empty:
        raise ValueError("Manifest has no rows with dat_exists=True.")
    return out.reset_index(drop=True)


def _attach_asassn_coordinates(
    df: pd.DataFrame,
    *,
    id_col: str,
    asassn_index: Path,
) -> pd.DataFrame:
    index_path = Path(asassn_index).expanduser()
    if not index_path.exists():
        raise FileNotFoundError(f"ASAS-SN index not found: {index_path}")

    available = _parquet_columns(index_path)
    required = {"asas_sn_id", "ra_deg", "dec_deg"}
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"ASAS-SN index is missing required columns: {', '.join(missing)}")

    optional = (
        "gaia_mag",
        "pstarrs_g_mag",
        "pstarrs_r_mag",
        "pstarrs_i_mag",
        "pstarrs_z_mag",
        "gaia_id",
        "refcat_id",
    )
    index_columns = ["asas_sn_id", "ra_deg", "dec_deg"]
    index_columns.extend(col for col in optional if col in available and col not in df.columns)
    index_df = read_parquet_table(index_path, columns=index_columns)
    index_df = index_df.copy()
    index_df["_asassn_join_id"] = _normalize_id_values(index_df["asas_sn_id"])
    index_df = index_df.loc[index_df["_asassn_join_id"].ne("")].drop_duplicates("_asassn_join_id", keep="last")

    work = df.drop(columns=[col for col in ("ra_deg", "dec_deg") if col in df.columns]).copy()
    work["_asassn_join_id"] = _normalize_id_values(work[id_col])
    join_columns = ["_asassn_join_id", *[col for col in index_columns if col != "asas_sn_id"]]
    if "asas_sn_id" not in work.columns:
        join_columns.append("asas_sn_id")

    out = work.merge(index_df[join_columns], on="_asassn_join_id", how="left", validate="many_to_one")
    out["asas_sn_id"] = out["_asassn_join_id"]
    out["ra_deg"] = pd.to_numeric(out["ra_deg"], errors="coerce")
    out["dec_deg"] = pd.to_numeric(out["dec_deg"], errors="coerce")
    has_coords = out["ra_deg"].notna() & out["dec_deg"].notna()
    out = out.loc[has_coords].drop(columns=["_asassn_join_id"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("No manifest rows could be joined to ASAS-SN index coordinates.")
    return out


def _resolve_dat_path(row: pd.Series, id_col: str) -> Path:
    if "dat_path" in row and pd.notna(row["dat_path"]):
        return Path(str(row["dat_path"])).expanduser()
    if "path" in row and pd.notna(row["path"]):
        path = Path(str(row["path"])).expanduser()
        if path.suffix == ".dat2":
            return path
    if "lc_dir" in row and pd.notna(row["lc_dir"]):
        return Path(str(row["lc_dir"])).expanduser() / f"{row[id_col]}.dat2"
    raise KeyError("Manifest row must provide dat_path, path, or lc_dir.")


def _series_value(row: pd.Series, *names: str, default: float = np.nan) -> float:
    for name in names:
        if name in row and pd.notna(row[name]):
            try:
                return float(row[name])
            except Exception:
                return float(default)
    return float(default)


def load_manifest(path: Path, *, asassn_index: Path | None = None) -> pd.DataFrame:
    df = read_parquet_table(path)
    id_col = _get_id_col(df)
    if not any(col in df.columns for col in ("dat_path", "path", "lc_dir")):
        raise ValueError("Manifest must include dat_path, path, or lc_dir.")
    df = _filter_existing_dat_rows(df)
    missing_coords = [col for col in ("ra_deg", "dec_deg") if col not in df.columns]
    if missing_coords:
        if asassn_index is None:
            raise ValueError(
                "Manifest missing required columns: "
                + ", ".join(missing_coords)
                + ". Pass --asassn-index to join coordinates from the ASAS-SN index."
            )
        df = _attach_asassn_coordinates(df, id_col=id_col, asassn_index=asassn_index)
    else:
        df = df.copy()
        df["ra_deg"] = pd.to_numeric(df["ra_deg"], errors="coerce")
        df["dec_deg"] = pd.to_numeric(df["dec_deg"], errors="coerce")
        df = df.loc[df["ra_deg"].notna() & df["dec_deg"].notna()].reset_index(drop=True)
        if df.empty:
            raise ValueError("Manifest has no rows with valid ra_deg/dec_deg coordinates.")
    return df


def select_control_sample(
    manifest_df: pd.DataFrame,
    *,
    n_sample: int,
    min_points: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    df = manifest_df.copy()
    if "n_points" in df.columns and min_points > 0:
        df = df[df["n_points"] >= min_points]
    if len(df) <= n_sample:
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(df), size=int(n_sample), replace=False)
    return df.iloc[pick].reset_index(drop=True)


def build_amplitude_grid(min_val: float, max_val: float, steps: int) -> np.ndarray:
    return np.linspace(float(min_val), float(max_val), int(steps))


def build_timescale_grid(min_val: float, max_val: float, steps: int) -> np.ndarray:
    return np.logspace(np.log10(float(min_val)), np.log10(float(max_val)), int(steps))


def parse_classes(value: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(value, str):
        parts = [part.strip().lower() for part in value.split(",")]
    else:
        parts = [str(part).strip().lower() for part in value]
    classes = [part for part in parts if part]
    invalid = sorted(set(classes) - set(TRUTH_CLASSES))
    if invalid:
        raise ValueError(f"Unsupported nuclear injection classes: {', '.join(invalid)}")
    return list(dict.fromkeys(classes))


def build_trial_specs(
    amplitude_values: np.ndarray,
    timescale_values: np.ndarray,
    *,
    repeats_per_grid: int,
    control_count: int,
    classes: list[str],
) -> list[TrialSpec]:
    specs: list[TrialSpec] = []
    trial_index = 0
    for truth_class in classes:
        if truth_class == "control":
            for source_index in range(int(control_count)):
                specs.append(
                    TrialSpec(
                        trial_index=trial_index,
                        truth_class="control",
                        amplitude_mag=math.nan,
                        timescale_days=math.nan,
                        repeat_index=0,
                        source_index=source_index,
                    )
                )
                trial_index += 1
            continue
        for amp in amplitude_values:
            for timescale in timescale_values:
                for repeat_index in range(int(repeats_per_grid)):
                    specs.append(
                        TrialSpec(
                            trial_index=trial_index,
                            truth_class=truth_class,
                            amplitude_mag=float(amp),
                            timescale_days=float(timescale),
                            repeat_index=int(repeat_index),
                        )
                    )
                    trial_index += 1
    return specs


def load_dat2_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        header=None,
        names=DAT2_COLUMNS,
        sep=r"\s+",
        dtype={
            "JD": "float64",
            "mag": "float64",
            "error": "float64",
            "good_bad": "int64",
            "camera": "string",
            "v_g_band": "int64",
            "saturated": "int64",
            "cam_field": "string",
        },
    )


def write_dat2_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=" ", header=False, index=False)


def _time_array(df_lc: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(df_lc["JD"], errors="coerce").to_numpy(dtype=float)


def inject_agn_variability(
    df_lc: pd.DataFrame,
    *,
    amplitude_mag: float,
    timescale_days: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    df_out = df_lc.copy()
    if df_out.empty:
        return df_out
    time = _time_array(df_out)
    order = np.argsort(time)
    sorted_time = time[order]
    tau = max(float(timescale_days), 1.0)
    state = np.zeros(len(sorted_time), dtype=float)
    state[0] = float(rng.normal())
    for idx in range(1, len(sorted_time)):
        dt = max(float(sorted_time[idx] - sorted_time[idx - 1]), 0.0)
        phi = float(np.exp(-dt / tau))
        state[idx] = phi * state[idx - 1] + math.sqrt(max(1.0 - phi * phi, 0.0)) * float(rng.normal())
    state = state - float(np.nanmedian(state))
    width = float(np.nanpercentile(state, 95) - np.nanpercentile(state, 5))
    if not math.isfinite(width) or width <= 0:
        width = float(np.nanstd(state))
    if not math.isfinite(width) or width <= 0:
        return df_out
    perturb_sorted = state / width * float(amplitude_mag)
    perturb = np.empty_like(perturb_sorted)
    perturb[order] = perturb_sorted
    df_out["mag"] = pd.to_numeric(df_out["mag"], errors="coerce").to_numpy(dtype=float) + perturb
    return df_out


def inject_tde_flare(
    df_lc: pd.DataFrame,
    *,
    amplitude_mag: float,
    timescale_days: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, float]:
    df_out = df_lc.copy()
    if df_out.empty:
        return df_out, math.nan
    time = _time_array(df_out)
    finite = time[np.isfinite(time)]
    if len(finite) == 0:
        return df_out, math.nan
    t_min = float(np.nanmin(finite))
    t_max = float(np.nanmax(finite))
    span = max(t_max - t_min, 1.0)
    low = t_min + min(max(100.0, 0.2 * span), 0.45 * span)
    high = t_max - min(max(100.0, 0.2 * span), 0.35 * span)
    if high <= low:
        low = t_min + 0.25 * span
        high = t_min + 0.75 * span
    peak_jd = float(rng.uniform(low, high)) if high > low else float(np.nanmedian(finite))
    decline_scale = max(float(timescale_days), 1.0)
    rise_scale = max(10.0, min(0.15 * decline_scale, 80.0))
    dt = time - peak_jd
    profile = np.where(
        dt < 0.0,
        np.exp(np.clip(dt / rise_scale, -20.0, 0.0)),
        np.power(1.0 + np.maximum(dt, 0.0) / decline_scale, -5.0 / 3.0),
    )
    profile[dt < -6.0 * rise_scale] = 0.0
    df_out["mag"] = pd.to_numeric(df_out["mag"], errors="coerce").to_numpy(dtype=float) - float(amplitude_mag) * profile
    return df_out, peak_jd - MJD_OFFSET


def inject_clagn_transition(
    df_lc: pd.DataFrame,
    *,
    amplitude_mag: float,
    timescale_days: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, int]:
    df_out = df_lc.copy()
    if df_out.empty:
        return df_out, 0
    time = _time_array(df_out)
    finite = time[np.isfinite(time)]
    if len(finite) == 0:
        return df_out, 0
    t_min = float(np.nanmin(finite))
    t_max = float(np.nanmax(finite))
    span = max(t_max - t_min, 1.0)
    center = float(rng.uniform(t_min + 0.3 * span, t_min + 0.7 * span))
    width = max(float(timescale_days) / 4.0, 1.0)
    direction = int(rng.choice([-1, 1]))
    step = 0.5 * (1.0 + np.tanh((time - center) / width))
    df_out["mag"] = pd.to_numeric(df_out["mag"], errors="coerce").to_numpy(dtype=float) + direction * float(amplitude_mag) * step
    return df_out, direction


def inject_nuclear_template(
    df_lc: pd.DataFrame,
    *,
    truth_class: str,
    amplitude_mag: float,
    timescale_days: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if truth_class == "control":
        return df_lc.copy(), {"injection_peak_mjd": math.nan, "injection_direction": 0}
    if truth_class == "agn":
        return (
            inject_agn_variability(
                df_lc,
                amplitude_mag=amplitude_mag,
                timescale_days=timescale_days,
                rng=rng,
            ),
            {"injection_peak_mjd": math.nan, "injection_direction": 0},
        )
    if truth_class == "tde":
        injected, peak_mjd = inject_tde_flare(
            df_lc,
            amplitude_mag=amplitude_mag,
            timescale_days=timescale_days,
            rng=rng,
        )
        return injected, {"injection_peak_mjd": peak_mjd, "injection_direction": -1}
    if truth_class == "clagn":
        injected, direction = inject_clagn_transition(
            df_lc,
            amplitude_mag=amplitude_mag,
            timescale_days=timescale_days,
            rng=rng,
        )
        return injected, {"injection_peak_mjd": math.nan, "injection_direction": direction}
    raise ValueError(f"Unsupported truth class: {truth_class}")


def synthetic_context_for_class(truth_class: str) -> dict[str, object]:
    base: dict[str, object] = {
        "parallax": 0.0,
        "parallax_error": 0.2,
        "pm_total": 0.0,
        "pmra_error": 1.0,
        "pmdec_error": 1.0,
        "host_nuclear_score": 1.0,
        "redshift": 0.05,
        "redshift_source": "injected",
        "injected_context_profile": truth_class,
    }
    if truth_class == "control":
        base.update(
            {
                "host_nuclear_score": 0.85,
                "w1": 13.0,
                "w2": 12.9,
                "w1_w2": 0.1,
                "radio_det": False,
                "xray_det": False,
                "swift_xrt_det": False,
                "swift_uvot_det": False,
                "simbad_otype": "Galaxy",
                "spectral_type": "Galaxy",
            }
        )
    elif truth_class == "agn":
        base.update(
            {
                "w1": 12.0,
                "w2": 11.0,
                "w1_w2": 1.0,
                "neowise_n_epochs": 5,
                "neowise_w1_range": 0.8,
                "neowise_w2_range": 0.6,
                "radio_det": True,
                "radio_flux_mjy": 25.0,
                "xray_det": True,
                "xray_flux": 2e-13,
                "simbad_otype": "QSO",
                "spectral_type": "QSO broad-line",
            }
        )
    elif truth_class == "tde":
        base.update(
            {
                "w1": 13.0,
                "w2": 12.85,
                "w1_w2": 0.15,
                "radio_det": False,
                "xray_det": False,
                "swift_xrt_det": False,
                "swift_uvot_obs": True,
                "swift_uvot_det": True,
                "galex_nuv": 19.0,
                "galex_fuv": 19.5,
                "phot_g_mean_mag": 18.6,
                "simbad_otype": "Galaxy",
                "spectral_type": "quiescent galaxy",
            }
        )
    elif truth_class == "clagn":
        base.update(
            {
                "w1": 12.0,
                "w2": 11.35,
                "w1_w2": 0.65,
                "neowise_n_epochs": 5,
                "neowise_w1_range": 0.6,
                "neowise_w2_range": 0.45,
                "radio_det": False,
                "xray_det": False,
                "broad_line_change_flag": True,
                "spectral_type": "changing-look nucleus",
            }
        )
    else:
        raise ValueError(f"Unsupported truth class: {truth_class}")
    return base


def _base_candidate_row(source_row: pd.Series, *, id_col: str, dat_path: Path) -> dict[str, object]:
    source_id = str(source_row[id_col])
    return {
        "candidate_id": source_id,
        "source_id": source_id,
        "asas_sn_id": source_id,
        "dat_path": str(dat_path),
        "ra_deg": _series_value(source_row, "ra_deg", "ra"),
        "dec_deg": _series_value(source_row, "dec_deg", "dec"),
        "pstarrs_g_mag": _series_value(source_row, "pstarrs_g_mag", "g_mag", "mag", default=np.nan),
    }


def score_injected_lightcurve(
    df_lc: pd.DataFrame,
    *,
    source_row: pd.Series,
    id_col: str,
    dat_path: Path,
    truth_class: str,
    peak_mjd: float | None,
    min_score: float,
    min_margin: float,
) -> pd.DataFrame:
    row = _base_candidate_row(source_row, id_col=id_col, dat_path=dat_path)
    row.update(compute_nuclear_lightcurve_features(df_lc, peak_mjd=peak_mjd))
    row.update(synthetic_context_for_class(truth_class))
    frame = pd.DataFrame([row])
    frame = resolve_redshift_spectral_types(frame)
    frame = score_nuclear_candidates(frame)
    return arbitrate_nuclear_scores(frame, min_score=min_score, min_margin=min_margin)


def _run_trial_worker(trial_index: int) -> dict:
    control_sample = _GLOBAL["control_sample"]
    specs = _GLOBAL["trial_specs"]
    seed = int(_GLOBAL["seed"])
    min_score = float(_GLOBAL["min_score"])
    min_margin = float(_GLOBAL["min_margin"])

    spec: TrialSpec = specs[trial_index]
    rng = np.random.default_rng(seed + int(trial_index))
    id_col = _get_id_col(control_sample)
    if spec.source_index is None:
        row_idx = int(rng.integers(len(control_sample)))
    else:
        row_idx = int(spec.source_index) % len(control_sample)
    source_row = control_sample.iloc[row_idx]
    dat_path = _resolve_dat_path(source_row, id_col)
    source_id = str(source_row[id_col])

    result = {
        "trial_index": int(trial_index),
        "source_id": source_id,
        "dat_path": str(dat_path),
        "truth_class": spec.truth_class,
        "amplitude_mag": float(spec.amplitude_mag) if math.isfinite(float(spec.amplitude_mag)) else np.nan,
        "timescale_days": float(spec.timescale_days) if math.isfinite(float(spec.timescale_days)) else np.nan,
        "repeat_index": int(spec.repeat_index),
        "source_row_index": int(row_idx),
        "pstarrs_g_mag": _series_value(source_row, "pstarrs_g_mag", "g_mag", "mag", default=np.nan),
        "ra_deg": _series_value(source_row, "ra_deg", "ra"),
        "dec_deg": _series_value(source_row, "dec_deg", "dec"),
        "n_points_raw": 0,
        "baseline_days": np.nan,
        "injection_peak_mjd": np.nan,
        "injection_direction": 0,
        "recovered": False,
        "ambiguous": False,
        "failure_reason": "error",
        "error": None,
    }

    try:
        df_raw = load_dat2_table(dat_path)
        result["n_points_raw"] = int(len(df_raw))
        result["baseline_days"] = float(df_raw["JD"].max() - df_raw["JD"].min()) if len(df_raw) else np.nan

        df_injected, injection_meta = inject_nuclear_template(
            df_raw,
            truth_class=spec.truth_class,
            amplitude_mag=float(spec.amplitude_mag) if math.isfinite(float(spec.amplitude_mag)) else 0.0,
            timescale_days=float(spec.timescale_days) if math.isfinite(float(spec.timescale_days)) else 1.0,
            rng=rng,
        )
        result.update(injection_meta)

        scored = score_injected_lightcurve(
            df_injected,
            source_row=source_row,
            id_col=id_col,
            dat_path=dat_path,
            truth_class=spec.truth_class,
            peak_mjd=result["injection_peak_mjd"],
            min_score=min_score,
            min_margin=min_margin,
        )
        scored_row = scored.iloc[0]
        for column in (
            *SCORE_COLUMNS,
            "gaia_stellar_veto_score",
            "gaia_extragalactic_prior_score",
            "wise_agn_score",
            "neowise_variability_score",
            "radio_agn_prior_score",
            "xray_agn_prior_score",
            "uv_tde_score",
            "nuc_n_points",
            "nuc_time_span_days",
            "nuc_flux_frac_amp_p95_p05",
            "nuc_flux_slope_snr",
            "n_flare_events",
            "recurrence_count",
            "preflare_rms",
            "tde_single_flare_score",
            "tde_quiet_baseline_score",
            "tde_no_recurrence_score",
            "tde_smooth_decline_score",
            "fallback_fit_r2",
            "clagn_state_change_mag",
            "clagn_monotonicity_score",
            "clagn_plateau_score",
            "host_nuclear_score",
            "w1_w2",
            "neowise_n_epochs",
            "neowise_w1_range",
            "radio_det",
            "xray_det",
            "swift_uvot_det",
            "swift_uvot_obs",
            "redshift",
            "redshift_source",
            "spectral_type",
            "host_spectral_class",
            "prior_agn_spectrum_flag",
            "broad_line_flag",
            "broad_line_change_flag",
            "injected_context_profile",
            "agn_prior_reasons",
            "tde_candidate_reasons",
            "clagn_reasons",
            "nuclear_primary_hypothesis",
            "nuclear_primary_score",
            "nuclear_best_score_hypothesis",
            "nuclear_best_score",
            "nuclear_runner_up_hypothesis",
            "nuclear_runner_up_score",
            "nuclear_hypothesis_margin",
            "nuclear_hypothesis_status",
        ):
            if column in scored_row:
                result[column] = scored_row[column]

        primary = str(result.get("nuclear_primary_hypothesis", ""))
        result["ambiguous"] = primary == "ambiguous" or result.get("nuclear_hypothesis_status") == "ambiguous"
        result["recovered"] = primary == spec.truth_class
        result["failure_reason"] = "recovered" if result["recovered"] else primary or "unknown"
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def _init_worker(
    control_sample: pd.DataFrame,
    trial_specs: list[TrialSpec],
    seed: int,
    min_score: float,
    min_margin: float,
) -> None:
    _GLOBAL["control_sample"] = control_sample
    _GLOBAL["trial_specs"] = trial_specs
    _GLOBAL["seed"] = int(seed)
    _GLOBAL["min_score"] = float(min_score)
    _GLOBAL["min_margin"] = float(min_margin)


def _run_trial_batch(trial_indices: list[int]) -> list[dict]:
    return [_run_trial_worker(trial_index) for trial_index in trial_indices]


def run_nuclear_injection_recovery(
    control_sample: pd.DataFrame,
    *,
    amplitude_values: np.ndarray,
    timescale_values: np.ndarray,
    repeats_per_grid: int,
    classes: list[str],
    seed: int,
    min_score: float = NUCLEAR_INJECTION_MIN_SCORE,
    min_margin: float = NUCLEAR_INJECTION_MIN_MARGIN,
    workers: int = 1,
    task_size: int = 50,
    checkpoint_interval: int = NUCLEAR_INJECTION_CHECKPOINT_INTERVAL,
    chunk_size: int = NUCLEAR_INJECTION_CHUNK_SIZE,
    output_path: Path | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = True,
    overwrite: bool = False,
    max_trials: int | None = None,
) -> pd.DataFrame | None:
    trial_specs = build_trial_specs(
        amplitude_values,
        timescale_values,
        repeats_per_grid=repeats_per_grid,
        control_count=len(control_sample) if "control" in classes else 0,
        classes=classes,
    )
    total_trials = len(trial_specs)
    if max_trials is not None:
        total_trials = min(total_trials, int(max_trials))
        trial_specs = trial_specs[:total_trials]

    if output_path is not None:
        output_path = Path(output_path)
        if output_path.exists() and overwrite and not resume:
            output_path.unlink()
        if output_path.exists() and not resume and not overwrite:
            raise SystemExit(f"Output exists: {output_path} (use --overwrite or --no-resume)")

    if checkpoint_path is None and output_path is not None:
        checkpoint_path = output_path.with_name(f"{output_path.stem}_PROCESSED.txt")

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists() and overwrite and not resume:
            checkpoint_path.unlink()

    start_index = 0
    if resume and checkpoint_path is not None and checkpoint_path.exists():
        last = _read_checkpoint(checkpoint_path)
        if last is not None:
            start_index = int(last) + 1

    if start_index >= total_trials:
        if output_path is not None and output_path.exists():
            return read_parquet_table(output_path)
        return pd.DataFrame()

    writer = ParquetAppendWriter(output_path) if output_path is not None else None
    results: list[dict] = []

    def flush_results() -> None:
        nonlocal results
        if not results:
            return
        if writer is not None:
            writer.write_chunk(results)
            results = []

    if workers <= 1:
        _init_worker(control_sample, trial_specs, seed, min_score, min_margin)
        for trial_index in range(start_index, total_trials):
            results.append(_run_trial_worker(trial_index))
            if chunk_size and len(results) >= chunk_size:
                flush_results()
            if checkpoint_path is not None and (trial_index + 1) % checkpoint_interval == 0:
                flush_results()
                _write_checkpoint(checkpoint_path, trial_index)
        flush_results()
        if checkpoint_path is not None:
            _write_checkpoint(checkpoint_path, total_trials - 1)
        if output_path is not None:
            return read_parquet_table(output_path)
        return pd.DataFrame(results)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(control_sample, trial_specs, seed, min_score, min_margin),
    ) as executor:
        for batch_start in range(start_index, total_trials, checkpoint_interval):
            batch_end = min(batch_start + checkpoint_interval, total_trials)
            batch_indices = list(range(batch_start, batch_end))
            tasks = [batch_indices[i : i + task_size] for i in range(0, len(batch_indices), task_size)]
            futures = {executor.submit(_run_trial_batch, task): task for task in tasks}
            for future in as_completed(futures):
                results.extend(future.result())
                if chunk_size and len(results) >= chunk_size:
                    flush_results()
            flush_results()
            if checkpoint_path is not None:
                _write_checkpoint(checkpoint_path, batch_end - 1)

    if output_path is not None:
        return read_parquet_table(output_path)
    return pd.DataFrame(results)


def compute_recovery_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame(columns=["truth_class", "nuclear_primary_hypothesis", "count", "fraction", "recovery_fraction"])
    grouped = (
        results_df.groupby(["truth_class", "nuclear_primary_hypothesis"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = grouped.groupby("truth_class")["count"].transform("sum")
    grouped["fraction"] = grouped["count"] / totals.where(totals > 0, np.nan)
    recovered = results_df.groupby("truth_class")["recovered"].mean().rename("recovery_fraction").reset_index()
    return grouped.merge(recovered, on="truth_class", how="left")


def compute_confusion_matrix(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()
    labels = list(TRUTH_CLASSES) + ["ambiguous"]
    matrix = pd.crosstab(
        results_df["truth_class"],
        results_df["nuclear_primary_hypothesis"],
        normalize="index",
    )
    matrix = matrix.reindex(index=list(TRUTH_CLASSES), columns=labels, fill_value=0.0)
    matrix.index.name = "truth_class"
    return matrix


def compute_fraction_grid(
    results_df: pd.DataFrame,
    *,
    value_column: str,
    amplitude_values: np.ndarray,
    timescale_values: np.ndarray,
) -> pd.DataFrame:
    pivot = results_df.pivot_table(
        index="amplitude_mag",
        columns="timescale_days",
        values=value_column,
        aggfunc="mean",
    )
    pivot = pivot.reindex(index=amplitude_values, columns=timescale_values)
    pivot.index.name = "amplitude_mag"
    return pivot


def compute_plot_tables(
    results_df: pd.DataFrame,
    *,
    amplitude_values: np.ndarray,
    timescale_values: np.ndarray,
    top_n_outcomes: int = 4,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    injected = results_df[results_df["truth_class"].isin(("agn", "tde", "clagn"))].copy()
    if injected.empty:
        return tables

    outcomes = (
        injected["nuclear_primary_hypothesis"]
        .fillna("unknown")
        .value_counts()
        .head(int(top_n_outcomes))
        .index.tolist()
    )
    for truth_class in ("agn", "tde", "clagn"):
        class_df = injected[injected["truth_class"] == truth_class].copy()
        if class_df.empty:
            continue
        class_df["recovery_fraction"] = class_df["recovered"].astype(float)
        tables[f"{truth_class}_recovery_fraction"] = compute_fraction_grid(
            class_df,
            value_column="recovery_fraction",
            amplitude_values=amplitude_values,
            timescale_values=timescale_values,
        )
        for outcome in outcomes:
            value_column = f"predicted_{outcome}_fraction"
            class_df[value_column] = (class_df["nuclear_primary_hypothesis"].fillna("unknown") == outcome).astype(float)
            tables[f"{truth_class}_predicted_{outcome}_fraction"] = compute_fraction_grid(
                class_df,
                value_column=value_column,
                amplitude_values=amplitude_values,
                timescale_values=timescale_values,
            )
    return tables


def _format_mag_slice_label(interval: pd.Interval) -> str:
    left = f"{float(interval.left):.2f}".replace(".", "p")
    right = f"{float(interval.right):.2f}".replace(".", "p")
    return f"gmag_{left}_{right}"


def compute_magnitude_slices(
    results_df: pd.DataFrame,
    *,
    mag_column: str = "pstarrs_g_mag",
    n_slices: int = 4,
) -> list[tuple[str, str, pd.DataFrame]]:
    if mag_column not in results_df.columns or n_slices <= 0:
        return []
    df = results_df.copy()
    valid = df[mag_column].notna()
    if valid.sum() < max(2, n_slices):
        return []
    try:
        bins = pd.qcut(df.loc[valid, mag_column], q=n_slices, duplicates="drop")
    except ValueError:
        return []
    if bins.empty:
        return []
    slices: list[tuple[str, str, pd.DataFrame]] = []
    for interval in bins.cat.categories:
        slice_index = bins[bins == interval].index
        if len(slice_index) == 0:
            continue
        label = _format_mag_slice_label(interval)
        display = f"{float(interval.left):.2f} <= g < {float(interval.right):.2f}"
        slices.append((label, display, df.loc[slice_index].copy()))
    return slices


def save_plot_tables(plot_tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in plot_tables.items():
        write_parquet_table(table.reset_index(), output_dir / f"{name}.parquet")


def _heatmap_edges(values: np.ndarray, *, log_scale: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        delta = values[0] * 0.1 if values[0] != 0 else 1.0
        return np.array([values[0] - delta, values[0] + delta], dtype=float)
    if log_scale:
        logs = np.log10(values)
        mid = (logs[:-1] + logs[1:]) / 2.0
        edges = np.empty(values.size + 1, dtype=float)
        edges[1:-1] = 10 ** mid
        edges[0] = 10 ** (logs[0] - (mid[0] - logs[0]))
        edges[-1] = 10 ** (logs[-1] + (logs[-1] - mid[-1]))
        return edges
    mid = (values[:-1] + values[1:]) / 2.0
    edges = np.empty(values.size + 1, dtype=float)
    edges[1:-1] = mid
    edges[0] = values[0] - (mid[0] - values[0])
    edges[-1] = values[-1] + (values[-1] - mid[-1])
    return edges


def plot_heatmap(
    grid_df: pd.DataFrame,
    *,
    output_path: Path,
    colorbar_label: str,
    cmap: str = "viridis",
    xlog: bool = True,
) -> plt.Figure:
    x_vals = np.asarray(grid_df.columns, dtype=float)
    y_vals = np.asarray(grid_df.index, dtype=float)
    z = grid_df.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_HEATMAP)
    mesh = ax.pcolormesh(
        _heatmap_edges(x_vals, log_scale=xlog),
        _heatmap_edges(y_vals, log_scale=False),
        z,
        shading="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
    )
    if xlog:
        ax.set_xscale("log")
    ax.set_xlabel("Injected Timescale [days]")
    ax.set_ylabel("Injected Amplitude [mag]")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(colorbar_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_publication_figure(fig, output_path, dpi=200, close=False)
    return fig


def plot_outcome_counts(summary_df: pd.DataFrame, *, output_path: Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_LC_WIDE)
    if summary_df.empty:
        ordered = pd.DataFrame({"outcome": [], "count": []})
    else:
        ordered = (
            summary_df.groupby("nuclear_primary_hypothesis")["count"]
            .sum()
            .sort_values(ascending=True)
            .rename_axis("outcome")
            .reset_index(name="count")
        )
    ax.barh(ordered["outcome"], ordered["count"], color="steelblue")
    ax.set_xlabel("Trials")
    ax.set_ylabel("Arbitrated Outcome")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_publication_figure(fig, output_path, dpi=200, close=False)
    return fig


def plot_confusion_matrix(matrix: pd.DataFrame, *, output_path: Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_HEATMAP)
    data = matrix.to_numpy(dtype=float) if not matrix.empty else np.zeros((0, 0))
    image = ax.imshow(data, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Arbitrated Hypothesis")
    ax.set_ylabel("Injected Truth")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Row Fraction")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_publication_figure(fig, output_path, dpi=200, close=False)
    return fig


def generate_plots(
    results_df: pd.DataFrame,
    *,
    amplitude_values: np.ndarray,
    timescale_values: np.ndarray,
    output_dir: Path,
    top_n_outcomes: int = 4,
    n_mag_slices: int = 4,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = compute_recovery_summary(results_df)
    fig = plot_outcome_counts(summary, output_path=output_dir / "arbitration_outcome_counts.png")
    plt.close(fig)

    matrix = compute_confusion_matrix(results_df)
    fig = plot_confusion_matrix(matrix, output_path=output_dir / "confusion_matrix.png")
    plt.close(fig)

    plot_tables = compute_plot_tables(
        results_df,
        amplitude_values=amplitude_values,
        timescale_values=timescale_values,
        top_n_outcomes=top_n_outcomes,
    )
    save_plot_tables(plot_tables, output_dir / "plot_tables")

    for name, table in plot_tables.items():
        if name.endswith("_recovery_fraction"):
            output_name = f"{name}_heatmap.png"
            label = "Recovery Fraction"
            cmap = "viridis"
        else:
            output_name = f"{name}_heatmap.png"
            label = "Prediction Fraction"
            cmap = "magma"
        fig = plot_heatmap(
            table,
            output_path=output_dir / output_name,
            colorbar_label=label,
            cmap=cmap,
        )
        plt.close(fig)

    mag_slices = compute_magnitude_slices(results_df, n_slices=n_mag_slices)
    if mag_slices:
        mag_slice_dir = output_dir / "magnitude_slices"
        mag_slice_tables_dir = output_dir / "plot_tables" / "magnitude_slices"
        for label, _display, slice_df in mag_slices:
            slice_tables = compute_plot_tables(
                slice_df,
                amplitude_values=amplitude_values,
                timescale_values=timescale_values,
                top_n_outcomes=top_n_outcomes,
            )
            save_plot_tables(slice_tables, mag_slice_tables_dir / label)
            for name, table in slice_tables.items():
                fig = plot_heatmap(
                    table,
                    output_path=mag_slice_dir / f"{label}_{name}_heatmap.png",
                    colorbar_label="Recovery Fraction" if name.endswith("_recovery_fraction") else "Prediction Fraction",
                    cmap="viridis" if name.endswith("_recovery_fraction") else "magma",
                )
                plt.close(fig)

    return plot_tables


def save_results_artifacts(
    results_df: pd.DataFrame,
    *,
    results_dir: Path,
    plot_tables: dict[str, pd.DataFrame] | None = None,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    write_parquet_table(results_df, results_dir / "nuclear_injection_trials.parquet")
    summary = compute_recovery_summary(results_df)
    write_parquet_table(summary, results_dir / "nuclear_recovery_summary.parquet")
    matrix = compute_confusion_matrix(results_df)
    write_parquet_table(matrix.reset_index(), results_dir / "nuclear_confusion_matrix.parquet")
    if plot_tables is not None:
        aggregate_dir = results_dir / "aggregates"
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        for name, table in plot_tables.items():
            write_parquet_table(table.reset_index(), aggregate_dir / f"{name}.parquet")


def _get_non_default_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    non_defaults = {}
    for action in parser._actions:
        if action.dest == "help":
            continue
        value = getattr(args, action.dest, None)
        if value != action.default:
            non_defaults[action.dest] = value
    return non_defaults


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run nuclear AGN/TDE/CLAGN injection-recovery arbitration tests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Output structure (default --output-dir {NUCLEAR_INJECTION_OUTPUT_DIR}):
  {NUCLEAR_INJECTION_OUTPUT_DIR}/
    20260314_101500/
      run_params.json
      results/
        nuclear_injection_trials.parquet
        nuclear_recovery_summary.parquet
        aggregates/
      plots/
        arbitration_outcome_counts.png
        confusion_matrix.png
        <class>_recovery_fraction_heatmap.png
        <class>_predicted_<hypothesis>_fraction_heatmap.png
        plot_tables/
        magnitude_slices/
    latest -> 20260314_101500/
""",
    )
    g_io = parser.add_argument_group("Input / output")
    g_sample = parser.add_argument_group("Sample")
    g_injection = parser.add_argument_group("Injection parameters")
    g_arbitration = parser.add_argument_group("Arbitration")
    g_workers = parser.add_argument_group("Workers & chunks")
    g_plots = parser.add_argument_group("Plots")

    g_io.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Manifest Parquet with dat_path/path/lc_dir; coordinates are joined from --asassn-index when absent.",
    )
    g_io.add_argument(
        "--asassn-index",
        type=Path,
        default=ASASSN_INDEX_PATH,
        help=f"ASAS-SN coordinate index used when the manifest lacks ra_deg/dec_deg (default: {ASASSN_INDEX_PATH}).",
    )
    g_io.add_argument("--output-dir", dest="out_dir", type=Path, default=NUCLEAR_INJECTION_OUTPUT_DIR, help=f"Base output directory (default: {NUCLEAR_INJECTION_OUTPUT_DIR}).")
    g_io.add_argument("--run-tag", type=str, default=None, help="Optional suffix for the run directory.")
    g_io.add_argument("--output", type=Path, default=None, help="Override trial Parquet output path.")

    g_sample.add_argument("--control-sample-size", type=int, default=NUCLEAR_INJECTION_CONTROL_SAMPLE_SIZE)
    g_sample.add_argument("--min-points", type=int, default=0, help="Optional n_points floor if present in the manifest.")
    g_sample.add_argument("--seed", type=int, default=0, help="Random seed for source and template draws.")

    g_injection.add_argument("--classes", type=str, default=NUCLEAR_INJECTION_CLASSES, help="Comma-separated classes to run.")
    g_injection.add_argument("--amp-min", type=float, default=NUCLEAR_INJECTION_AMP_MIN)
    g_injection.add_argument("--amp-max", type=float, default=NUCLEAR_INJECTION_AMP_MAX)
    g_injection.add_argument("--amp-steps", type=int, default=NUCLEAR_INJECTION_AMP_STEPS)
    g_injection.add_argument("--timescale-min", type=float, default=NUCLEAR_INJECTION_TIMESCALE_MIN_DAYS)
    g_injection.add_argument("--timescale-max", type=float, default=NUCLEAR_INJECTION_TIMESCALE_MAX_DAYS)
    g_injection.add_argument("--timescale-steps", type=int, default=NUCLEAR_INJECTION_TIMESCALE_STEPS)
    g_injection.add_argument("--repeats-per-grid", type=int, default=NUCLEAR_INJECTION_REPEATS_PER_GRID)

    g_arbitration.add_argument("--min-score", type=float, default=NUCLEAR_INJECTION_MIN_SCORE)
    g_arbitration.add_argument("--min-margin", type=float, default=NUCLEAR_INJECTION_MIN_MARGIN)

    g_workers.add_argument("--workers", type=int, default=1, help="Parallel workers.")
    g_workers.add_argument("--task-size", type=int, default=25, help="Trials per worker task.")
    g_workers.add_argument("--checkpoint-interval", type=int, default=NUCLEAR_INJECTION_CHECKPOINT_INTERVAL)
    g_workers.add_argument("--chunk-size", type=int, default=NUCLEAR_INJECTION_CHUNK_SIZE)
    g_workers.add_argument("--max-trials", type=int, default=None, help="Optional debug cap on total trials.")
    g_workers.add_argument("--no-resume", action="store_true", help="Disable resume mode.")
    g_workers.add_argument("--overwrite", action="store_true", help="Overwrite output/checkpoint when not resuming.")

    g_plots.add_argument("--top-outcomes", type=int, default=4, help="Number of arbitrated outcomes to plot as heatmaps.")
    g_plots.add_argument("--mag-slices", type=int, default=4, help="Number of magnitude slices for sliced heatmaps.")
    g_plots.add_argument("--skip-plots", action="store_true", help="Skip figure generation.")

    args = parser.parse_args()

    classes = parse_classes(args.classes)
    if not classes:
        raise SystemExit("No nuclear injection classes requested.")
    amplitude_values = build_amplitude_grid(args.amp_min, args.amp_max, args.amp_steps)
    timescale_values = build_timescale_grid(args.timescale_min, args.timescale_max, args.timescale_steps)

    base_out_dir = Path(args.out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{args.run_tag}" if args.run_tag else timestamp
    run_dir = base_out_dir / run_name
    results_dir = run_dir / "results"
    plots_dir = run_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    results_out = args.output if args.output is not None else (results_dir / "nuclear_injection_trials.parquet")
    checkpoint_path = results_out.with_name(f"{results_out.stem}_PROCESSED.txt")

    manifest_df = load_manifest(Path(args.manifest), asassn_index=Path(args.asassn_index))
    control_sample = select_control_sample(
        manifest_df,
        n_sample=int(args.control_sample_size),
        min_points=int(args.min_points),
        seed=int(args.seed),
    )
    if control_sample.empty:
        raise SystemExit("Control sample is empty after filtering.")

    run_params = {
        key: _jsonable(value)
        for key, value in _get_non_default_args(args, parser).items()
    }
    run_params["classes"] = classes
    run_params["amplitude_values"] = amplitude_values.tolist()
    run_params["timescale_values"] = timescale_values.tolist()
    run_params["control_sample_rows"] = int(len(control_sample))
    (run_dir / "run_params.json").write_text(json.dumps(run_params, indent=2), encoding="ascii")

    results_df = run_nuclear_injection_recovery(
        control_sample,
        amplitude_values=amplitude_values,
        timescale_values=timescale_values,
        repeats_per_grid=int(args.repeats_per_grid),
        classes=classes,
        seed=int(args.seed),
        min_score=float(args.min_score),
        min_margin=float(args.min_margin),
        workers=int(args.workers),
        task_size=int(args.task_size),
        checkpoint_interval=int(args.checkpoint_interval),
        chunk_size=int(args.chunk_size),
        output_path=results_out,
        checkpoint_path=checkpoint_path,
        resume=not args.no_resume,
        overwrite=bool(args.overwrite),
        max_trials=args.max_trials,
    )
    if results_df is None:
        results_df = read_parquet_table(results_out)

    plot_tables = None
    if not args.skip_plots:
        plot_tables = generate_plots(
            results_df,
            amplitude_values=amplitude_values,
            timescale_values=timescale_values,
            output_dir=plots_dir,
            top_n_outcomes=int(args.top_outcomes),
            n_mag_slices=int(args.mag_slices),
        )
    save_results_artifacts(results_df, results_dir=results_dir, plot_tables=plot_tables)

    latest_link = base_out_dir / "latest"
    try:
        if latest_link.is_symlink() or latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(run_dir.name)
    except OSError:
        pass

    summary = compute_recovery_summary(results_df)
    print(f"Run directory: {run_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
