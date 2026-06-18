"""
Injection-recovery style rejection benchmark for the LTV pipeline.

This module injects smooth long-term trends into real ASAS-SN light curves,
re-runs the production LTV core metric extraction, then records whether each
trial survives the existing LTV filter stack or is rejected by a specific
filter. The primary outputs are plots showing pass fraction and rejection
fractions across injected-parameter space.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from malca.lightcurve_publication import (
    apply_publication_rcparams,
    FIG_SINGLE_COL_HEATMAP,
    FIG_SINGLE_COL_LC_WIDE,
    FIG_TWO_COL_STANDARD,
)

apply_publication_rcparams(plt)

from malca.config import (
    LTV_CHUNK_SIZE,
    LTV_DSPRING,
    LTV_INJECTION_AMP_MAX,
    LTV_INJECTION_AMP_MIN,
    LTV_INJECTION_AMP_STEPS,
    LTV_INJECTION_CHECKPOINT_INTERVAL,
    LTV_INJECTION_CHUNK_SIZE,
    LTV_INJECTION_CONTROL_SAMPLE_SIZE,
    LTV_INJECTION_PROFILE,
    LTV_INJECTION_REPEATS_PER_GRID,
    LTV_INJECTION_TIMESCALE_MAX_DAYS,
    LTV_INJECTION_TIMESCALE_MIN_DAYS,
    LTV_INJECTION_TIMESCALE_STEPS,
    LTV_MAX_SEASONS,
    LTV_MIN_DIFF,
    LTV_MIN_POINTS_PER_SEASON,
    LTV_MIN_SEASONS_FOR_QUADRATIC,
    LTV_MIN_SLOPE,
    LTV_WORKERS,
)
from malca.config import LTV_INJECTION_OUTPUT_DIR
from malca.ltv.core import Config, SourceMeta, process_one_lc
from malca.ltv.filter import apply_all_filters
from malca.table_io import read_parquet_table, write_parquet_table


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

_GLOBAL: dict[str, object] = {}


@dataclass(frozen=True)
class TrialSpec:
    trial_index: int
    amplitude_mag: float
    timescale_days: float
    direction: int


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
        df_chunk = df_chunk.reindex(columns=self.columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size > 0:
            existing = read_parquet_table(self.path)
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


def _load_table(path: Path) -> pd.DataFrame:
    return read_parquet_table(path)


def _get_id_col(df: pd.DataFrame) -> str:
    for col in ("asas_sn_id", "ASAS-SN ID", "source_id", "id"):
        if col in df.columns:
            return col
    raise KeyError("Manifest is missing a usable ID column.")


def _resolve_dat_path(row: pd.Series, id_col: str) -> Path:
    if "dat_path" in row and pd.notna(row["dat_path"]):
        return Path(str(row["dat_path"]))
    if "path" in row and pd.notna(row["path"]):
        path = Path(str(row["path"]))
        if path.suffix == ".dat2":
            return path
    if "lc_dir" in row and pd.notna(row["lc_dir"]):
        return Path(str(row["lc_dir"])) / f"{row[id_col]}.dat2"
    raise KeyError("Manifest row must provide dat_path, path, or lc_dir.")


def load_manifest(path: Path) -> pd.DataFrame:
    df = _load_table(path)
    id_col = _get_id_col(df)
    required = {id_col, "ra_deg", "dec_deg", "pstarrs_g_mag"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Manifest missing required columns: {', '.join(missing)}")
    if not any(col in df.columns for col in ("dat_path", "path", "lc_dir")):
        raise ValueError("Manifest must include dat_path, path, or lc_dir.")
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
    pick = rng.choice(len(df), size=n_sample, replace=False)
    return df.iloc[pick].reset_index(drop=True)


def build_amplitude_grid(min_val: float, max_val: float, steps: int) -> np.ndarray:
    return np.linspace(float(min_val), float(max_val), int(steps))


def build_timescale_grid(min_val: float, max_val: float, steps: int) -> np.ndarray:
    return np.logspace(np.log10(float(min_val)), np.log10(float(max_val)), int(steps))


def build_trial_specs(
    amplitude_values: np.ndarray,
    timescale_values: np.ndarray,
    *,
    repeats_per_grid: int,
    seed: int,
    direction_mode: str,
) -> list[TrialSpec]:
    rng = np.random.default_rng(seed)
    specs: list[TrialSpec] = []
    trial_index = 0
    for amp in amplitude_values:
        for timescale in timescale_values:
            for _ in range(int(repeats_per_grid)):
                if direction_mode == "positive":
                    direction = 1
                elif direction_mode == "negative":
                    direction = -1
                else:
                    direction = int(rng.choice([-1, 1]))
                specs.append(
                    TrialSpec(
                        trial_index=trial_index,
                        amplitude_mag=float(amp),
                        timescale_days=float(timescale),
                        direction=direction,
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


def inject_trend(
    df_lc: pd.DataFrame,
    *,
    amplitude_mag: float,
    timescale_days: float,
    direction: int = 1,
    profile: str = "tanh",
) -> pd.DataFrame:
    df_out = df_lc.copy()
    if df_out.empty:
        return df_out

    t = df_out["JD"].to_numpy(dtype=float)
    center = float(np.median(t))
    direction = 1 if int(direction) >= 0 else -1

    if profile == "linear":
        span = max(float(t.max() - t.min()), 1.0)
        trend = direction * float(amplitude_mag) * ((t - center) / span)
    elif profile == "tanh":
        scale = max(float(timescale_days), 1.0)
        trend = 0.5 * direction * float(amplitude_mag) * np.tanh((t - center) / scale)
    else:
        raise ValueError(f"Unsupported profile: {profile}")

    df_out["mag"] = df_out["mag"].to_numpy(dtype=float) + trend
    return df_out


def build_ltv_config(args: argparse.Namespace) -> Config:
    return Config(
        root=Path("."),
        mag_bin="injection",
        output=Path("."),
        dspring=float(args.dspring),
        ra_is_deg=bool(args.ra_is_deg),
        max_seasons=int(args.max_seasons),
        min_points_per_season=int(args.min_points_per_season),
        min_seasons_for_quadratic=int(args.min_seasons_for_quadratic),
        write_per_dir=False,
        band_mode=str(args.band_mode),
        workers=1,
        chunk_size=1,
        overwrite=False,
    )


def _series_value(row: pd.Series, *names: str, default: float = np.nan) -> float:
    for name in names:
        if name in row and pd.notna(row[name]):
            return float(row[name])
    return float(default)


def _run_trial_worker(trial_index: int) -> dict:
    control_sample = _GLOBAL["control_sample"]
    specs = _GLOBAL["trial_specs"]
    seed = int(_GLOBAL["seed"])
    profile = str(_GLOBAL["profile"])
    cfg = _GLOBAL["cfg"]
    filter_kwargs = _GLOBAL["filter_kwargs"]

    spec = specs[trial_index]
    rng = np.random.default_rng(seed + int(trial_index))
    row_idx = int(rng.integers(len(control_sample)))
    source_row = control_sample.iloc[row_idx]
    id_col = _get_id_col(control_sample)
    source_id = str(source_row[id_col])
    dat_path = _resolve_dat_path(source_row, id_col)

    result = {
        "trial_index": int(trial_index),
        "source_id": source_id,
        "dat_path": str(dat_path),
        "profile": profile,
        "amplitude_mag": float(spec.amplitude_mag),
        "timescale_days": float(spec.timescale_days),
        "direction": int(spec.direction),
        "pstarrs_g_mag": _series_value(source_row, "pstarrs_g_mag"),
        "ra_deg": _series_value(source_row, "ra_deg"),
        "dec_deg": _series_value(source_row, "dec_deg"),
        "passed": False,
        "filter_reason": "error",
        "error": None,
    }

    try:
        df_raw = load_dat2_table(dat_path)
        result["n_points_raw"] = int(len(df_raw))
        result["baseline_days"] = float(df_raw["JD"].max() - df_raw["JD"].min()) if len(df_raw) else np.nan

        df_injected = inject_trend(
            df_raw,
            amplitude_mag=float(spec.amplitude_mag),
            timescale_days=float(spec.timescale_days),
            direction=int(spec.direction),
            profile=profile,
        )

        with tempfile.TemporaryDirectory(prefix="ltv_injection_") as tmpdir:
            temp_path = Path(tmpdir) / dat_path.name
            write_dat2_table(df_injected, temp_path)

            meta = SourceMeta(
                asas_sn_id=int(float(source_row[id_col])),
                ra_deg=float(source_row["ra_deg"]),
                dec_deg=float(source_row["dec_deg"]),
                pstarrs_g_mag=float(source_row["pstarrs_g_mag"]),
            )
            ltv_row = process_one_lc(str(temp_path), meta, cfg)

        if ltv_row is None:
            result["filter_reason"] = "core_no_metrics"
            return result

        result.update(
            {
                "measured_slope": _series_value(pd.Series(ltv_row), "ltv_slope"),
                "measured_max_diff": _series_value(pd.Series(ltv_row), "ltv_max_diff"),
                "measured_median": _series_value(pd.Series(ltv_row), "ltv_median"),
                "measured_ls_fap": _series_value(pd.Series(ltv_row), "ltv_ls_fap"),
            }
        )

        df_metrics = pd.DataFrame([ltv_row])
        filter_output = apply_all_filters(
            df_metrics,
            return_rejected=True,
            **filter_kwargs,
        )
        passed_df, rejected_df = filter_output

        if not passed_df.empty:
            result["passed"] = True
            result["filter_reason"] = "passed"
        elif rejected_df is not None and not rejected_df.empty:
            result["filter_reason"] = str(rejected_df.iloc[0].get("filter_reason", "filtered_out"))
        else:
            result["filter_reason"] = "filtered_out"
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def _init_worker(
    control_sample: pd.DataFrame,
    trial_specs: list[TrialSpec],
    cfg: Config,
    filter_kwargs: dict,
    profile: str,
    seed: int,
) -> None:
    _GLOBAL["control_sample"] = control_sample
    _GLOBAL["trial_specs"] = trial_specs
    _GLOBAL["cfg"] = cfg
    _GLOBAL["filter_kwargs"] = filter_kwargs
    _GLOBAL["profile"] = profile
    _GLOBAL["seed"] = int(seed)


def _run_trial_batch(trial_indices: list[int]) -> list[dict]:
    return [_run_trial_worker(trial_index) for trial_index in trial_indices]


def compute_rejection_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame(columns=["filter_reason", "count", "fraction"])
    counts = results_df["filter_reason"].fillna("unknown").value_counts(dropna=False)
    total = int(counts.sum())
    summary = counts.rename_axis("filter_reason").reset_index(name="count")
    summary["fraction"] = summary["count"] / total if total > 0 else 0.0
    return summary


def compute_fraction_grid(
    results_df: pd.DataFrame,
    *,
    value_column: str,
    amplitude_values: np.ndarray,
    timescale_values: np.ndarray,
) -> pd.DataFrame:
    df = results_df.copy()
    pivot = df.pivot_table(
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
    top_n_reasons: int = 4,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    pass_df = results_df.copy()
    pass_df["pass_fraction"] = pass_df["passed"].astype(float)
    tables["pass_fraction"] = compute_fraction_grid(
        pass_df,
        value_column="pass_fraction",
        amplitude_values=amplitude_values,
        timescale_values=timescale_values,
    )

    summary = compute_rejection_summary(results_df)
    rejection_reasons = [
        reason
        for reason in summary["filter_reason"].tolist()
        if reason not in {"passed", "error"}
    ][:top_n_reasons]

    for reason in rejection_reasons:
        reason_df = results_df.copy()
        reason_df[f"{reason}_fraction"] = (reason_df["filter_reason"] == reason).astype(float)
        tables[f"rejection_{reason}"] = compute_fraction_grid(
            reason_df,
            value_column=f"{reason}_fraction",
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

    df.loc[valid, "_mag_slice"] = bins.astype(str)
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
    title: str,
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
    ax.set_xlabel("Injected Timescale (days)")
    ax.set_ylabel("Injected Amplitude (mag)")
    ax.set_title(title)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_reason_breakdown(summary_df: pd.DataFrame, *, output_path: Path) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIG_SINGLE_COL_LC_WIDE)
    ordered = summary_df.sort_values("count", ascending=True)
    ax.barh(ordered["filter_reason"], ordered["count"], color="steelblue")
    ax.set_xlabel("Trials")
    ax.set_ylabel("Outcome")
    ax.set_title("LTV Injection Outcomes by First Rejection Reason")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def generate_plots(
    results_df: pd.DataFrame,
    *,
    amplitude_values: np.ndarray,
    timescale_values: np.ndarray,
    output_dir: Path,
    top_n_reasons: int = 4,
    n_mag_slices: int = 4,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = compute_rejection_summary(results_df)
    fig = plot_reason_breakdown(summary, output_path=output_dir / "rejection_reason_counts.png")
    plt.close(fig)

    plot_tables = compute_plot_tables(
        results_df,
        amplitude_values=amplitude_values,
        timescale_values=timescale_values,
        top_n_reasons=top_n_reasons,
    )
    save_plot_tables(plot_tables, output_dir / "plot_tables")

    pass_grid = plot_tables["pass_fraction"]
    fig = plot_heatmap(
        pass_grid,
        title="LTV Pass Fraction Across Injected Trend Space",
        output_path=output_dir / "pass_fraction_heatmap.png",
        colorbar_label="Pass Fraction",
        cmap="viridis",
    )
    plt.close(fig)

    for name, table in plot_tables.items():
        if name == "pass_fraction":
            continue
        reason = name.replace("rejection_", "")
        fig = plot_heatmap(
            table,
            title=f"Fraction Rejected by {reason}",
            output_path=output_dir / f"{name}_heatmap.png",
            colorbar_label="Rejection Fraction",
            cmap="magma",
        )
        plt.close(fig)

    mag_slices = compute_magnitude_slices(results_df, n_slices=n_mag_slices)
    if mag_slices:
        mag_slice_dir = output_dir / "magnitude_slices"
        mag_slice_tables_dir = output_dir / "plot_tables" / "magnitude_slices"
        for label, display, slice_df in mag_slices:
            slice_plot_tables = compute_plot_tables(
                slice_df,
                amplitude_values=amplitude_values,
                timescale_values=timescale_values,
                top_n_reasons=top_n_reasons,
            )
            save_plot_tables(slice_plot_tables, mag_slice_tables_dir / label)

            fig = plot_heatmap(
                slice_plot_tables["pass_fraction"],
                title=f"LTV Pass Fraction ({display})",
                output_path=mag_slice_dir / f"{label}_pass_fraction_heatmap.png",
                colorbar_label="Pass Fraction",
                cmap="viridis",
            )
            plt.close(fig)

            for name, table in slice_plot_tables.items():
                if name == "pass_fraction":
                    continue
                reason = name.replace("rejection_", "")
                fig = plot_heatmap(
                    table,
                    title=f"Fraction Rejected by {reason} ({display})",
                    output_path=mag_slice_dir / f"{label}_{name}_heatmap.png",
                    colorbar_label="Rejection Fraction",
                    cmap="magma",
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
    write_parquet_table(results_df, results_dir / "ltv_injection_trials.parquet")

    summary = compute_rejection_summary(results_df)
    write_parquet_table(summary, results_dir / "ltv_rejection_summary.parquet")

    if plot_tables is not None:
        aggregate_dir = results_dir / "aggregates"
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        for name, table in plot_tables.items():
            write_parquet_table(table.reset_index(), aggregate_dir / f"{name}.parquet")


def run_injection_recovery(
    control_sample: pd.DataFrame,
    *,
    amplitude_values: np.ndarray,
    timescale_values: np.ndarray,
    repeats_per_grid: int,
    profile: str,
    direction_mode: str,
    cfg: Config,
    filter_kwargs: dict,
    seed: int,
    workers: int = 1,
    task_size: int = 50,
    checkpoint_interval: int = LTV_INJECTION_CHECKPOINT_INTERVAL,
    chunk_size: int = LTV_INJECTION_CHUNK_SIZE,
    output_path: Path | None = None,
    checkpoint_path: Path | None = None,
    resume: bool = True,
    overwrite: bool = False,
    max_trials: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame | None:
    trial_specs = build_trial_specs(
        amplitude_values,
        timescale_values,
        repeats_per_grid=repeats_per_grid,
        seed=seed,
        direction_mode=direction_mode,
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
        _init_worker(control_sample, trial_specs, cfg, filter_kwargs, profile, seed)
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
        initargs=(control_sample, trial_specs, cfg, filter_kwargs, profile, seed),
    ) as executor:
        for batch_start in range(start_index, total_trials, checkpoint_interval):
            batch_end = min(batch_start + checkpoint_interval, total_trials)
            batch_indices = list(range(batch_start, batch_end))
            tasks = [batch_indices[i:i + task_size] for i in range(0, len(batch_indices), task_size)]
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
        description="Run LTV rejection-recovery injections and generate diagnostic plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Output structure (default --output-dir {LTV_INJECTION_OUTPUT_DIR}):
  {LTV_INJECTION_OUTPUT_DIR}/
    20260314_101500/
      run_params.json
      results/
        ltv_injection_trials.parquet
        ltv_rejection_summary.parquet
        aggregates/
      plots/
        rejection_reason_counts.png
        pass_fraction_heatmap.png
        rejection_<reason>_heatmap.png
        plot_tables/
        magnitude_slices/
    latest -> 20260314_101500/
""",
    )
    g_io = parser.add_argument_group("Input / output")
    g_sample = parser.add_argument_group("Sample")
    g_injection = parser.add_argument_group("Injection parameters")
    g_workers = parser.add_argument_group("Workers & chunks")
    g_ltv = parser.add_argument_group("LTV core")
    g_filter = parser.add_argument_group("Filter")
    g_plots = parser.add_argument_group("Plots")

    g_io.add_argument("--manifest", type=Path, required=True, help="Manifest Parquet with dat_path metadata.")
    g_io.add_argument("--output-dir", dest="out_dir", type=Path, default=LTV_INJECTION_OUTPUT_DIR, help=f"Base output directory (default: {LTV_INJECTION_OUTPUT_DIR}).")
    g_io.add_argument("--run-tag", type=str, default=None, help="Optional suffix for the run directory.")
    g_io.add_argument("--output", type=Path, default=None, help="Override trial Parquet output path.")
    g_sample.add_argument(
        "--control-sample-size",
        type=int,
        default=LTV_INJECTION_CONTROL_SAMPLE_SIZE,
        help="Number of control light curves to sample from the manifest.",
    )
    g_sample.add_argument("--min-points", type=int, default=0, help="Optional n_points floor if present in the manifest.")
    g_sample.add_argument("--seed", type=int, default=0, help="Random seed for source selection and direction draws.")

    g_injection.add_argument("--profile", type=str, default=LTV_INJECTION_PROFILE, choices=["tanh", "linear"])
    g_injection.add_argument("--direction-mode", type=str, default="both", choices=["both", "positive", "negative"])
    g_injection.add_argument("--amp-min", type=float, default=LTV_INJECTION_AMP_MIN)
    g_injection.add_argument("--amp-max", type=float, default=LTV_INJECTION_AMP_MAX)
    g_injection.add_argument("--amp-steps", type=int, default=LTV_INJECTION_AMP_STEPS)
    g_injection.add_argument("--timescale-min", type=float, default=LTV_INJECTION_TIMESCALE_MIN_DAYS)
    g_injection.add_argument("--timescale-max", type=float, default=LTV_INJECTION_TIMESCALE_MAX_DAYS)
    g_injection.add_argument("--timescale-steps", type=int, default=LTV_INJECTION_TIMESCALE_STEPS)
    g_injection.add_argument("--repeats-per-grid", type=int, default=LTV_INJECTION_REPEATS_PER_GRID)

    g_workers.add_argument("--workers", type=int, default=1, help="Parallel workers.")
    g_workers.add_argument("--task-size", type=int, default=25, help="Trials per worker task.")
    g_workers.add_argument("--checkpoint-interval", type=int, default=LTV_INJECTION_CHECKPOINT_INTERVAL)
    g_workers.add_argument("--chunk-size", type=int, default=LTV_INJECTION_CHUNK_SIZE)
    g_workers.add_argument("--max-trials", type=int, default=None, help="Optional debug cap on total trials.")
    g_workers.add_argument("--no-resume", action="store_true", help="Disable resume mode.")
    g_workers.add_argument("--overwrite", action="store_true", help="Overwrite output/checkpoint when not resuming.")

    g_ltv.add_argument("--band-mode", type=str, default="pipeline", choices=["pipeline", "g_only"])
    g_ltv.add_argument("--dspring", type=float, default=LTV_DSPRING)
    g_ltv.add_argument(
        "--ra-is-deg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Interpret manifest RA as degrees when computing seasonal midpoints.",
    )
    g_ltv.add_argument("--max-seasons", type=int, default=LTV_MAX_SEASONS)
    g_ltv.add_argument("--min-points-per-season", type=int, default=LTV_MIN_POINTS_PER_SEASON)
    g_ltv.add_argument("--min-seasons-for-quadratic", type=int, default=LTV_MIN_SEASONS_FOR_QUADRATIC)

    g_filter.add_argument("--min-slope", type=float, default=LTV_MIN_SLOPE)
    g_filter.add_argument("--min-diff", type=float, default=LTV_MIN_DIFF)
    g_filter.add_argument("--query-gaia", action=argparse.BooleanOptionalAction, default=True)
    g_filter.add_argument("--run-enhanced-filters", action=argparse.BooleanOptionalAction, default=True)
    g_filter.add_argument("--run-neighbor-pm-filter", action=argparse.BooleanOptionalAction, default=True)
    g_filter.add_argument("--filter-chunk-size", type=int, default=LTV_CHUNK_SIZE)
    g_filter.add_argument("--filter-workers", type=int, default=LTV_WORKERS)

    g_plots.add_argument("--top-reasons", type=int, default=4, help="Number of rejection reasons to plot as heatmaps.")
    g_plots.add_argument("--mag-slices", type=int, default=4, help="Number of magnitude slices for sliced heatmaps.")
    g_plots.add_argument("--skip-plots", action="store_true", help="Skip figure generation.")

    args = parser.parse_args()

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

    results_out = args.output if args.output is not None else (results_dir / "ltv_injection_trials.parquet")
    checkpoint_path = results_out.with_name(f"{results_out.stem}_PROCESSED.txt")

    manifest_df = load_manifest(Path(args.manifest))
    control_sample = select_control_sample(
        manifest_df,
        n_sample=args.control_sample_size,
        min_points=args.min_points,
        seed=args.seed,
    )
    if control_sample.empty:
        raise SystemExit("Control sample is empty after filtering.")

    cfg = build_ltv_config(args)
    filter_kwargs = dict(
        min_slope=float(args.min_slope),
        min_diff=float(args.min_diff),
        query_gaia=bool(args.query_gaia),
        run_enhanced_filters=bool(args.run_enhanced_filters),
        run_neighbor_pm_filter=bool(args.run_neighbor_pm_filter),
        chunk_size=int(args.filter_chunk_size),
        n_workers=int(args.filter_workers),
        verbose=False,
        log_csv=None,
    )

    run_params = {
        key: _jsonable(value)
        for key, value in _get_non_default_args(args, parser).items()
    }
    run_params["amplitude_values"] = amplitude_values.tolist()
    run_params["timescale_values"] = timescale_values.tolist()
    run_params["control_sample_rows"] = int(len(control_sample))
    (run_dir / "run_params.json").write_text(json.dumps(run_params, indent=2), encoding="ascii")

    results_df = run_injection_recovery(
        control_sample,
        amplitude_values=amplitude_values,
        timescale_values=timescale_values,
        repeats_per_grid=int(args.repeats_per_grid),
        profile=str(args.profile),
        direction_mode=str(args.direction_mode),
        cfg=cfg,
        filter_kwargs=filter_kwargs,
        seed=int(args.seed),
        workers=int(args.workers),
        task_size=int(args.task_size),
        checkpoint_interval=int(args.checkpoint_interval),
        chunk_size=int(args.chunk_size),
        output_path=results_out,
        checkpoint_path=checkpoint_path,
        resume=not args.no_resume,
        overwrite=bool(args.overwrite),
        max_trials=args.max_trials,
        show_progress=True,
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
            top_n_reasons=int(args.top_reasons),
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

    summary = compute_rejection_summary(results_df)
    print(f"Run directory: {run_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
