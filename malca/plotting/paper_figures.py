"""Publication figure generators for MALCA paper Tier A/B plots."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import patches
from matplotlib.ticker import AutoMinorLocator

from malca.evaluation.dip_injection import load_efficiency_grid_depth_timescale
from malca.io.notebook_paths import find_repo_root
from malca.ltv.cmd import dustmaps_cmd_from_fields
from malca.plotting.color_color_labels import (
    LABEL_KS_W3,
    LABEL_KS_W4,
    LABEL_KS_W3_0,
    LABEL_KS_W4_0,
    LABEL_R_HALPHA,
    color_color_mag_label,
    dereddened_color_color_mag_label,
)
from malca.plotting.extinction import add_dereddened_ir_magnitudes, dereddened_color
from malca.plotting.lightcurve_publication import (
    FIG_SINGLE_COL_WIDTH,
    PUBLICATION_STYLE,
    apply_publication_rcparams,
    save_publication_figure,
)
from malca.plotting.notebook_display import show_figure
from malca.review.filter_schema import VETTING_KNOWN_SELECT_FILTERS, is_dipper_contaminant_type_value
from malca.review.store import get_distinct_values, query_queue

Q_PERIODIC_BOUNDARY = 0.60
Q_QUASI_PERIODIC_BOUNDARY = 0.80

DISK_BOUNDARY_SEGMENTS = {
    "diskless_debris": [(0.00, 0.50), (0.42, -0.32)],
    "debris_evolved": [(1.50, 1.25), (2.40, -0.25)],
    "evolved_full": [(2.50, 2.50), (3.50, 1.50), (5.00, 0.00)],
    "evolved_transition": [(3.50, 1.50), (6.55, 2.70)],
}


@dataclass
class PaperFigureContext:
    repo_root: Path
    run_root: Path
    review_db: Path
    output_dir: Path
    sed_dir: Path = field(init=False)
    injection_recovery_path: Path | None = None
    efficiency_cube_path: Path | None = None  # backwards-compatible alias
    export_pdf: bool = True
    export_png: bool = True
    show_inline: bool = False

    def __post_init__(self) -> None:
        self.sed_dir = self.run_root / "results" / "marked_dipper_seds"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        explicit = self.injection_recovery_path or self.efficiency_cube_path
        if explicit is not None and not explicit.exists():
            explicit = None
        resolved = resolve_injection_recovery_path(
            self.repo_root,
            self.run_root,
            explicit=explicit,
        )
        self.injection_recovery_path = resolved
        self.efficiency_cube_path = resolved if resolved is not None and resolved.suffix == ".npz" else None


FigureResult = tuple[list[Path], plt.Figure]


def _candidate_manifest_paths(run_root: Path, repo_root: Path) -> list[Path]:
    candidates = [
        run_root / "results" / "external_lc_manifest.parquet",
        run_root / "results" / "lc_manifest_all.parquet",
        run_root / "lc_manifest_all.parquet",
        repo_root / "output" / "lc_manifest_all.parquet",
    ]
    return [path.resolve() for path in candidates if path.exists()]


def _manifest_from_run_params(run_dir: Path, repo_root: Path) -> Path | None:
    params_path = run_dir / "run_params.json"
    if not params_path.exists():
        return None
    try:
        data = json.loads(params_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    manifest_text = data.get("manifest")
    if not manifest_text:
        return None
    manifest = Path(str(manifest_text)).expanduser()
    if not manifest.is_absolute():
        manifest = (repo_root / manifest).resolve()
    else:
        manifest = manifest.resolve()
    return manifest


def _manifest_matches(injection_manifest: Path | None, run_manifest: Path) -> bool:
    if injection_manifest is None:
        return False
    try:
        return injection_manifest.samefile(run_manifest)
    except OSError:
        return injection_manifest == run_manifest.resolve()


def _injection_run_dir_for_artifact(artifact_path: Path) -> Path | None:
    if artifact_path.parent.name == "cubes":
        return artifact_path.parent.parent
    if artifact_path.parent.name == "results":
        return artifact_path.parent.parent
    return artifact_path.parent


def _collect_injection_artifacts(
    repo_root: Path,
    run_root: Path,
    *,
    filename: str,
) -> list[Path]:
    run_manifests = _candidate_manifest_paths(run_root, repo_root)
    search_roots = [
        run_root / "injection",
        run_root / "results" / "injection",
        run_root / "dip_injection",
        repo_root / "output" / "dip_injection",
        repo_root / "output" / "injection",
        repo_root / "output_migrated_camera_field_20260606" / "injection",
    ]

    artifact_paths: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        latest_candidate = root / "latest" / "results" / filename
        if latest_candidate.exists():
            artifact_paths.append(latest_candidate.resolve())
        artifact_paths.extend(
            path.resolve()
            for path in root.rglob(filename)
            if path.is_file()
        )

    seen: set[str] = set()
    unique_artifacts: list[Path] = []
    for path in artifact_paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique_artifacts.append(path)

    if not unique_artifacts:
        return []

    def _score(artifact_path: Path) -> tuple[int, float]:
        run_dir = _injection_run_dir_for_artifact(artifact_path)
        injection_manifest = (
            _manifest_from_run_params(run_dir, repo_root) if run_dir is not None else None
        )
        manifest_match = bool(
            run_manifests
            and injection_manifest is not None
            and any(_manifest_matches(injection_manifest, manifest) for manifest in run_manifests)
        )
        return (1 if manifest_match else 0, artifact_path.stat().st_mtime)

    unique_artifacts.sort(key=_score, reverse=True)
    return unique_artifacts


def resolve_injection_results_path(
    repo_root: Path,
    run_root: Path,
    *,
    explicit: Path | str | None = None,
) -> Path | None:
    """Locate the newest ``injection_results.parquet`` for this run."""
    if explicit is not None:
        explicit_path = Path(explicit).expanduser()
        if explicit_path.exists() and explicit_path.suffix == ".parquet":
            return explicit_path.resolve()

    env_text = os.environ.get("MALCA_INJECTION_RESULTS", "").strip()
    if env_text:
        env_path = Path(env_text).expanduser()
        if env_path.exists():
            return env_path.resolve()

    artifacts = _collect_injection_artifacts(
        repo_root,
        run_root,
        filename="injection_results.parquet",
    )
    return artifacts[0] if artifacts else None


def resolve_efficiency_cube_path(
    repo_root: Path,
    run_root: Path,
    *,
    explicit: Path | str | None = None,
) -> Path | None:
    """Locate the newest injection-recovery efficiency cube for this run."""
    if explicit is not None:
        explicit_path = Path(explicit).expanduser()
        if explicit_path.exists() and explicit_path.suffix == ".npz":
            return explicit_path.resolve()

    env_text = os.environ.get("MALCA_EFFICIENCY_CUBE", "").strip()
    if env_text:
        env_path = Path(env_text).expanduser()
        if env_path.exists():
            return env_path.resolve()

    artifacts = _collect_injection_artifacts(
        repo_root,
        run_root,
        filename="efficiency_cube.npz",
    )
    return artifacts[0] if artifacts else None


def resolve_injection_recovery_path(
    repo_root: Path,
    run_root: Path,
    *,
    explicit: Path | str | None = None,
) -> Path | None:
    """Locate injection-recovery output for depth–timescale completeness overlays.

    Prefers ``injection_results.parquet`` (2D grid computed on the fly) and falls
    back to ``efficiency_cube.npz`` when only the cube is available.
    """
    if explicit is not None:
        explicit_path = Path(explicit).expanduser()
        if explicit_path.exists():
            return explicit_path.resolve()

    for env_name in ("MALCA_INJECTION_RESULTS", "MALCA_EFFICIENCY_CUBE"):
        env_text = os.environ.get(env_name, "").strip()
        if env_text:
            env_path = Path(env_text).expanduser()
            if env_path.exists():
                return env_path.resolve()

    return resolve_injection_results_path(repo_root, run_root) or resolve_efficiency_cube_path(
        repo_root,
        run_root,
    )


def default_context(
    run_name: str = "dat3-full-extended_2026-07-01-v4",
    output_subdir: str = "paper_figures",
) -> PaperFigureContext:
    repo_root = find_repo_root()
    run_root = repo_root / "output" / "runs" / run_name
    return PaperFigureContext(
        repo_root=repo_root,
        run_root=run_root,
        review_db=run_root / "review" / "review.db",
        output_dir=repo_root / "output" / "notebooks" / output_subdir,
    )


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)


def _save_figure(
    ctx: PaperFigureContext,
    fig: plt.Figure,
    stem: str,
) -> FigureResult:
    written: list[Path] = []
    if ctx.export_pdf:
        pdf_path = ctx.output_dir / f"{stem}.pdf"
        save_publication_figure(fig, pdf_path, close=False)
        written.append(pdf_path)
    if ctx.export_png:
        png_path = ctx.output_dir / f"{stem}.png"
        fig.savefig(png_path, dpi=300, bbox_inches=None)
        written.append(png_path)
    if not ctx.show_inline:
        plt.close(fig)
    return written, fig


def _existing_candidate_columns(conn: sqlite3.Connection, columns: list[str]) -> list[str]:
    available = {row[1] for row in conn.execute("PRAGMA table_info(candidates)")}
    return [col for col in columns if col in available]


def load_survey_candidates(review_db: Path) -> pd.DataFrame:
    """All candidates with variability stats and optional review labels."""
    desired = [
        "candidate_id",
        "stats_amplitude",
        "stats_photometry_median_mag",
        "stats_error_and_snr_stats_error_median",
        "stats_variability_quasi_periodicity_q",
        "stats_variability_flux_asymmetry_m",
        "stats_variability_periodic_feature_period_source",
        "ra",
        "dec",
        "gal_l",
        "gal_b",
        "bp_rp",
        "phot_g_mean_mag",
        "A_v_3d",
        "age50",
        "period_consensus_days",
        "period_primary_source",
        "tmass_k",
        "tmass_k_err",
        "w1",
        "w1_err",
        "w2",
        "w2_err",
        "w3",
        "w3_err",
        "w4",
        "w4_err",
        "sed_alpha",
        "iphas_r_ha",
        "vphas_r_ha",
        "pmra",
        "pmdec",
        "parallax",
        "distance_gspphot",
        "dip_best_mag_event",
        "dip_max_run_duration",
        "allwise_mep_w1_range",
        "stats_intrinsic_sigma_mag",
    ]
    with _connect_readonly(review_db) as conn:
        present = _existing_candidate_columns(conn, desired)
        if "candidate_id" not in present:
            raise RuntimeError(f"candidates table in {review_db} has no candidate_id column")
        select_cols = ", ".join(f"c.{col}" for col in present)
        query = f"""
            SELECT
                {select_cols},
                r.event_class,
                r.status AS review_status
            FROM candidates c
            LEFT JOIN reviews r ON r.candidate_id = c.candidate_id
        """
        frame = pd.read_sql_query(query, conn)
    numeric_cols = [c for c in frame.columns if c not in {"candidate_id", "event_class", "review_status", "period_primary_source", "stats_variability_periodic_feature_period_source"}]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["is_dipper"] = (
        frame["event_class"].fillna("").astype(str).str.strip().str.lower().eq("dipper")
    )
    return frame


def load_dippers(review_db: Path) -> pd.DataFrame:
    survey = load_survey_candidates(review_db)
    dippers = survey.loc[survey["is_dipper"]].copy()
    if dippers.empty:
        raise RuntimeError(f"No dippers found in {review_db}")
    return dippers


def _median_g(frame: pd.DataFrame) -> pd.Series:
    if "stats_photometry_median_mag" in frame.columns:
        return pd.to_numeric(frame["stats_photometry_median_mag"], errors="coerce")
    return pd.to_numeric(frame.get("phot_g_mean_mag"), errors="coerce")


def _column_as_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def _deredden_ir_colors(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_dereddened_ir_magnitudes(frame)
    out["ks_w2_0"] = dereddened_color(out, "tmass_k", "w2")
    out["ks_w3_0"] = dereddened_color(out, "tmass_k", "w3")
    out["ks_w4_0"] = dereddened_color(out, "tmass_k", "w4")
    return out


def _style_axis(ax: plt.Axes) -> None:
    ax.tick_params(direction="in", top=True, right=True, which="both")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


def plot_amplitude_vs_median_g(ctx: PaperFigureContext, frame: pd.DataFrame | None = None) -> FigureResult:
    frame = load_survey_candidates(ctx.review_db) if frame is None else frame
    x = _median_g(frame)
    y = pd.to_numeric(frame["stats_amplitude"], errors="coerce")
    good = np.isfinite(x) & np.isfinite(y) & (y >= 0)
    plot_df = frame.loc[good].copy()
    plot_df["median_g"] = x[good]
    plot_df["amplitude"] = y[good]

    fig, ax = plt.subplots(figsize=(FIG_SINGLE_COL_WIDTH, FIG_SINGLE_COL_WIDTH * 0.85), layout="constrained")
    background = plot_df.loc[~plot_df["is_dipper"]]
    dippers = plot_df.loc[plot_df["is_dipper"]]
    ax.scatter(background["median_g"], background["amplitude"], s=6, c="0.55", alpha=0.18, rasterized=True, label="Survey")
    ax.scatter(dippers["median_g"], dippers["amplitude"], s=28, c="#d85c1c", edgecolors="black", linewidths=0.25, alpha=0.9, zorder=4, label=rf"Dippers ($N={len(dippers)}$)")
    ax.set_xlabel(r"Median $g$ [mag]")
    ax.set_ylabel(r"Variability amplitude [mag]")
    ax.legend(frameon=False, fontsize=8)
    _style_axis(ax)
    return _save_figure(ctx, fig, "tier_a_amplitude_vs_median_g")


def plot_error_vs_median_g(ctx: PaperFigureContext, frame: pd.DataFrame | None = None) -> FigureResult:
    frame = load_survey_candidates(ctx.review_db) if frame is None else frame
    x = _median_g(frame)
    y = pd.to_numeric(frame["stats_error_and_snr_stats_error_median"], errors="coerce")
    good = np.isfinite(x) & np.isfinite(y) & (y > 0)
    plot_df = frame.loc[good].copy()
    plot_df["median_g"] = x[good]
    plot_df["error_median"] = y[good]

    fig, ax = plt.subplots(figsize=(FIG_SINGLE_COL_WIDTH, FIG_SINGLE_COL_WIDTH * 0.85), layout="constrained")
    hb = ax.hexbin(
        plot_df.loc[~plot_df["is_dipper"], "median_g"],
        plot_df.loc[~plot_df["is_dipper"], "error_median"],
        gridsize=45,
        cmap="Greys",
        mincnt=1,
        linewidths=0.0,
        alpha=0.55,
    )
    dippers = plot_df.loc[plot_df["is_dipper"]]
    ax.scatter(dippers["median_g"], dippers["error_median"], s=24, c="#2166ac", edgecolors="white", linewidths=0.2, alpha=0.85, zorder=4)
    ax.set_xlabel(r"Median $g$ [mag]")
    ax.set_ylabel(r"Photometric error median [mag]")
    _style_axis(ax)
    return _save_figure(ctx, fig, "tier_a_error_vs_median_g")


def _q_factor(source: object) -> float:
    match = re.search(r":q_factor_([0-9.]+)", str(source or ""))
    if match is None:
        return 1.0
    try:
        return float(match.group(1))
    except ValueError:
        return 1.0


def _q_class(q: float) -> str:
    if q < Q_PERIODIC_BOUNDARY:
        return "Periodic"
    if q < Q_QUASI_PERIODIC_BOUNDARY:
        return "Quasi-periodic"
    return "Aperiodic"


def _m_class(m: float) -> str:
    if m < -0.25:
        return "Bursting"
    if m <= 0.25:
        return "Symmetric"
    return "Dipping"


def plot_mq_diagram(ctx: PaperFigureContext, frame: pd.DataFrame | None = None) -> FigureResult:
    frame = load_survey_candidates(ctx.review_db) if frame is None else frame
    with _connect_readonly(ctx.review_db) as conn:
        exclusion_filters: dict[str, object] = {
            "select_filter_mode": "exclude",
            "nearby_vsx_dipper_contaminant_mode": "False",
        }
        for col in VETTING_KNOWN_SELECT_FILTERS:
            values = [
                str(value)
                for value in get_distinct_values(conn, col)
                if is_dipper_contaminant_type_value(col, value)
            ]
            if values:
                exclusion_filters[f"exclude_{col}"] = values
        queue_ids = set(query_queue(conn, filters=exclusion_filters, ids_only=True)["candidate_id"].astype(str))

    finite = frame.loc[
        frame["candidate_id"].isin(queue_ids)
        & np.isfinite(frame["stats_variability_quasi_periodicity_q"])
        & np.isfinite(frame["stats_variability_flux_asymmetry_m"])
    ].copy()
    finite["q"] = finite["stats_variability_quasi_periodicity_q"]
    finite["m_plot"] = finite["stats_variability_flux_asymmetry_m"].clip(-1.0, 1.0)

    fig, ax = plt.subplots(figsize=(7.0, 7.8), layout="constrained")
    background = finite.loc[~finite["is_dipper"]]
    dippers = finite.loc[finite["is_dipper"]]
    ax.scatter(background["q"], background["m_plot"], s=8, c="0.42", alpha=0.22, rasterized=True)
    ax.scatter(dippers["q"], dippers["m_plot"], s=42, c="#d85c1c", edgecolors="black", linewidths=0.25, alpha=0.95, zorder=5)
    for x in (Q_PERIODIC_BOUNDARY, Q_QUASI_PERIODIC_BOUNDARY):
        ax.axvline(x, color="black", lw=0.65, ls="--", zorder=0)
    for y in (-0.25, 0.25):
        ax.axhline(y, color="black", lw=0.65, ls="--", zorder=0)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, -1.0)
    ax.set_xlabel(r"Quasi-periodicity ($Q$)")
    ax.set_ylabel(r"Flux asymmetry ($M$)")
    _style_axis(ax)
    return _save_figure(ctx, fig, "tier_a_mq_diagram")


def plot_dip_depth_timescale(
    ctx: PaperFigureContext,
    dippers: pd.DataFrame | None = None,
    *,
    overlay_efficiency: bool = True,
) -> FigureResult:
    dippers = load_dippers(ctx.review_db) if dippers is None else dippers.copy()
    depth_mag = pd.to_numeric(dippers["dip_best_mag_event"], errors="coerce")
    duration = pd.to_numeric(dippers["dip_max_run_duration"], errors="coerce")
    fractional_depth = 1.0 - np.power(10.0, -0.4 * depth_mag)
    good = (
        np.isfinite(fractional_depth)
        & fractional_depth.gt(0)
        & fractional_depth.lt(1)
        & np.isfinite(duration)
        & duration.gt(0)
    )
    plot_df = dippers.loc[good].copy()
    plot_df["fractional_depth"] = fractional_depth[good]
    plot_df["timescale_days"] = duration[good]

    fig, ax = plt.subplots(figsize=(7.0, 6.5), layout="constrained")
    if overlay_efficiency and ctx.injection_recovery_path and ctx.injection_recovery_path.exists():
        dur_centers, depth_centers, eff = load_efficiency_grid_depth_timescale(
            ctx.injection_recovery_path
        )
        ax.pcolormesh(
            dur_centers,
            depth_centers,
            eff,
            cmap="magma",
            alpha=0.35,
            shading="auto",
            vmin=0.0,
            vmax=1.0,
            zorder=0,
        )
    ax.scatter(
        plot_df["timescale_days"],
        plot_df["fractional_depth"],
        s=48,
        c="#455a54",
        edgecolors="#202724",
        linewidths=0.6,
        alpha=0.82,
        zorder=3,
        label=rf"MALCA dippers ($N={len(plot_df)}$)",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"Dip timescale [days]")
    ax.set_ylabel(r"Fractional dip depth")
    ax.legend(frameon=False)
    _style_axis(ax)
    return _save_figure(ctx, fig, "tier_a_dip_depth_timescale")


def plot_ir_excess_vs_bprp(ctx: PaperFigureContext, dippers: pd.DataFrame | None = None) -> FigureResult:
    dippers = load_dippers(ctx.review_db) if dippers is None else dippers
    summary_path = ctx.sed_dir / "marked_dipper_sed_excess_summary.csv"
    controls_path = ctx.sed_dir / "sed_excess_null_controls.parquet"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing SED excess summary: {summary_path}")
    summary = pd.read_csv(summary_path)
    summary["candidate_id"] = summary["candidate_id"].astype(str)
    dip = dippers.merge(summary, on="candidate_id", how="inner", suffixes=("", "_sed"))
    dip["log_w4_ratio"] = np.log10(pd.to_numeric(dip["w4_ratio_p50"], errors="coerce").clip(lower=1e-3))

    controls = pd.DataFrame()
    if controls_path.exists():
        controls = pd.read_parquet(controls_path)
        controls["candidate_id"] = controls["candidate_id"].astype(str)
        if "band" in controls.columns:
            controls = controls.loc[controls["band"].astype(str).str.upper().eq("W4")].copy()
        controls["log_w4_ratio"] = np.log10(pd.to_numeric(controls.get("ratio_p50"), errors="coerce").clip(lower=1e-3))
        controls["bp_rp"] = pd.to_numeric(controls.get("bp_rp"), errors="coerce")

    fig, ax = plt.subplots(figsize=(FIG_SINGLE_COL_WIDTH * 1.4, FIG_SINGLE_COL_WIDTH), layout="constrained")
    if not controls.empty:
        good_c = np.isfinite(controls["bp_rp"]) & np.isfinite(controls["log_w4_ratio"])
        ax.scatter(
            controls.loc[good_c, "bp_rp"],
            controls.loc[good_c, "log_w4_ratio"],
            s=8,
            c="0.65",
            alpha=0.15,
            rasterized=True,
            label="Null controls",
        )
    good_d = np.isfinite(dip["bp_rp"]) & np.isfinite(dip["log_w4_ratio"])
    ax.scatter(
        dip.loc[good_d, "bp_rp"],
        dip.loc[good_d, "log_w4_ratio"],
        s=36,
        c="#b2182b",
        edgecolors="black",
        linewidths=0.25,
        alpha=0.9,
        zorder=4,
        label=rf"Dippers ($N={good_d.sum()}$)",
    )
    ax.set_xlabel(r"$G_{\mathrm{BP}}-G_{\mathrm{RP}}$")
    ax.set_ylabel(r"$\log_{10}(F_{\mathrm{W4}}/F_{\mathrm{photosphere}})$")
    ax.legend(frameon=False, fontsize=8)
    _style_axis(ax)
    return _save_figure(ctx, fig, "tier_a_ir_excess_vs_bprp")


def _plot_disk_color_color(
    ctx: PaperFigureContext,
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    stem: str,
    *,
    xlabel: str,
    ylabel: str,
) -> FigureResult:
    plotted = frame.dropna(subset=[x_col, y_col]).copy()
    fig, ax = plt.subplots(figsize=(7.0, 6.5), layout="constrained")
    for segment, points in DISK_BOUNDARY_SEGMENTS.items():
        xs, ys = zip(*points)
        ax.plot(xs, ys, color="0.35", lw=0.8, ls="--", zorder=1)
    ax.scatter(plotted[x_col], plotted[y_col], s=42, c="#2166ac", edgecolors="black", linewidths=0.35, alpha=0.88, zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _style_axis(ax)
    return _save_figure(ctx, fig, stem)


def plot_ks_w2_w4_dered(ctx: PaperFigureContext, dippers: pd.DataFrame | None = None) -> FigureResult:
    dippers = _deredden_ir_colors(load_dippers(ctx.review_db) if dippers is None else dippers)
    return _plot_disk_color_color(
        ctx,
        dippers,
        "ks_w2_0",
        "ks_w4_0",
        "tier_a_ks_w2_w4_dered",
        xlabel=dereddened_color_color_mag_label(r"K_s", r"W_2", "Ks", "W2"),
        ylabel=LABEL_KS_W4_0,
    )


def plot_ks_w3_w4_observed(ctx: PaperFigureContext, dippers: pd.DataFrame | None = None) -> FigureResult:
    dippers = load_dippers(ctx.review_db) if dippers is None else dippers.copy()
    dippers["ks_w3"] = pd.to_numeric(dippers["tmass_k"], errors="coerce") - pd.to_numeric(dippers["w3"], errors="coerce")
    dippers["ks_w4"] = pd.to_numeric(dippers["tmass_k"], errors="coerce") - pd.to_numeric(dippers["w4"], errors="coerce")
    return _plot_disk_color_color(
        ctx,
        dippers,
        "ks_w4",
        "ks_w3",
        "tier_b_ks_w3_w4_observed",
        xlabel=LABEL_KS_W4,
        ylabel=LABEL_KS_W3,
    )


def plot_extinction_corrected_cmd(ctx: PaperFigureContext, dippers: pd.DataFrame | None = None) -> FigureResult:
    """Render a dereddened Gaia CMD for reviewed dippers."""
    dippers = load_dippers(ctx.review_db) if dippers is None else dippers
    rows = []
    for _, row in dippers.iterrows():
        cmd = dustmaps_cmd_from_fields(
            g_mag=row.get("phot_g_mean_mag"),
            bp_rp=row.get("bp_rp"),
            a_v_3d=row.get("A_v_3d"),
            parallax_mas=row.get("parallax"),
            dist_pc=row.get("distance_gspphot"),
        )
        if cmd.get("bprp0") is None or cmd.get("mg0") is None:
            continue
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "bp_rp0": cmd["bprp0"],
                "abs_g0": cmd["mg0"],
            }
        )
    cmd_df = pd.DataFrame(rows)
    if cmd_df.empty:
        raise RuntimeError("No dereddened CMD points for dippers")

    fig, ax = plt.subplots(figsize=(FIG_SINGLE_COL_WIDTH * 1.2, FIG_SINGLE_COL_WIDTH * 1.2), layout="constrained")
    ax.scatter(cmd_df["bp_rp0"], cmd_df["abs_g0"], s=36, c="#2166ac", edgecolors="black", linewidths=0.3, alpha=0.9)
    ax.set_xlabel(r"$(G_{\mathrm{BP}}-G_{\mathrm{RP}})_0$")
    ax.set_ylabel(r"$M_{G,0}$")
    ax.invert_yaxis()
    _style_axis(ax)
    return _save_figure(ctx, fig, "tier_a_extinction_corrected_cmd")


def plot_representative_seds(
    ctx: PaperFigureContext,
    *,
    n_per_class: int = 1,
) -> FigureResult:
    residuals_path = ctx.sed_dir / "marked_dipper_sed_point_residuals.parquet"
    summary_path = ctx.sed_dir / "marked_dipper_sed_summary.csv"
    if not residuals_path.exists() or not summary_path.exists():
        raise FileNotFoundError("Run july1_marked_dipper_seds notebook or malca sed-excess first.")
    residuals = pd.read_parquet(residuals_path)
    summary = pd.read_csv(summary_path)
    summary["candidate_id"] = summary["candidate_id"].astype(str)
    class_col = "sed_alpha_class" if "sed_alpha_class" in summary.columns else "excess_class"
    picks: list[str] = []
    for cls, group in summary.groupby(summary[class_col].fillna("unknown")):
        ordered = group.sort_values("reduced_chi2" if "reduced_chi2" in group.columns else class_col)
        picks.extend(ordered["candidate_id"].head(n_per_class).astype(str).tolist())
    picks = picks[: min(len(picks), 6)]

    ncols = min(3, len(picks))
    nrows = int(np.ceil(len(picks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_SINGLE_COL_WIDTH * ncols, FIG_SINGLE_COL_WIDTH * 0.9 * nrows), layout="constrained", squeeze=False)
    lsun = 3.828e33
    for ax, cid in zip(axes.ravel(), picks):
        pts = residuals.loc[residuals["candidate_id"].astype(str).eq(cid)].copy()
        x = pd.to_numeric(pts["lambda_eff_angstrom"], errors="coerce")
        y = pd.to_numeric(pts["lambda_l_lambda"], errors="coerce") / lsun
        good = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        ax.scatter(x[good], y[good], s=18, c="#2166ac", edgecolors="black", linewidths=0.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(str(cid), fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(picks) :]:
        ax.axis("off")
    fig.supxlabel(r"$\lambda$ [Å]")
    fig.supylabel(r"$\lambda L_\lambda$ [$L_\odot$]")
    return _save_figure(ctx, fig, "tier_a_representative_sed_decompositions")


def plot_sed_with_residuals(ctx: PaperFigureContext, candidate_id: str | None = None) -> FigureResult:
    residuals_path = ctx.sed_dir / "marked_dipper_sed_point_residuals.parquet"
    summary_path = ctx.sed_dir / "marked_dipper_sed_summary.csv"
    if not residuals_path.exists():
        raise FileNotFoundError(residuals_path)
    residuals = pd.read_parquet(residuals_path)
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    if candidate_id is None and not summary.empty:
        candidate_id = str(summary.sort_values("reduced_chi2").iloc[0]["candidate_id"])
    candidate_id = str(candidate_id)
    pts = residuals.loc[residuals["candidate_id"].astype(str).eq(candidate_id)].copy()
    if pts.empty:
        raise ValueError(f"No SED residuals for {candidate_id}")

    lsun = 3.828e33
    x = pd.to_numeric(pts["lambda_eff_angstrom"], errors="coerce")
    y = pd.to_numeric(pts["lambda_l_lambda"], errors="coerce") / lsun
    ratio = _column_as_series(pts, "ratio_p50")
    if not np.isfinite(ratio).any():
        ratio = _column_as_series(pts, "model_ratio")
    good = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy()) & (x.to_numpy() > 0) & (y.to_numpy() > 0)
    good_arr = np.asarray(good, dtype=bool)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(FIG_SINGLE_COL_WIDTH * 1.2, FIG_SINGLE_COL_WIDTH * 1.35), sharex=True, layout="constrained", height_ratios=[3, 1])
    ax_top.scatter(x[good_arr], y[good_arr], s=24, c="#2166ac", edgecolors="black", linewidths=0.25, zorder=3)
    ratio_good = good_arr & np.isfinite(ratio.to_numpy())
    if ratio_good.any():
        model_y = y.to_numpy()[ratio_good] / ratio.to_numpy()[ratio_good]
        ax_top.plot(x.to_numpy()[ratio_good], model_y, color="black", lw=1.0, zorder=2, label="Photosphere")
    ax_top.set_yscale("log")
    ax_top.set_ylabel(r"$\lambda L_\lambda$ [$L_\odot$]")
    ax_top.legend(frameon=False, fontsize=8)

    resid = ratio.to_numpy()[good_arr] - 1.0
    ax_bot.axhline(0.0, color="0.3", lw=0.6)
    ax_bot.scatter(x.to_numpy()[good_arr], resid, s=18, c="#b2182b", edgecolors="black", linewidths=0.2)
    ax_bot.set_xscale("log")
    ax_bot.set_xlabel(r"$\lambda$ [Å]")
    ax_bot.set_ylabel(r"$F/F_* - 1$")
    fig.suptitle(candidate_id, fontsize=9, y=1.02)
    return _save_figure(ctx, fig, f"tier_a_sed_residuals_{candidate_id}")


def plot_sed_alpha_vs_halpha(ctx: PaperFigureContext, dippers: pd.DataFrame | None = None) -> FigureResult:
    dippers = load_dippers(ctx.review_db) if dippers is None else dippers
    alpha = pd.to_numeric(dippers["sed_alpha"], errors="coerce")
    ha = pd.to_numeric(dippers["iphas_r_ha"].fillna(dippers["vphas_r_ha"]), errors="coerce")
    good = np.isfinite(alpha) & np.isfinite(ha)
    fig, ax = plt.subplots(figsize=(FIG_SINGLE_COL_WIDTH, FIG_SINGLE_COL_WIDTH * 0.85), layout="constrained")
    ax.scatter(ha[good], alpha[good], s=34, c="#5f8f79", edgecolors="black", linewidths=0.25)
    ax.set_xlabel(LABEL_R_HALPHA + " (proxy for H$\\alpha$ EW)")
    ax.set_ylabel(r"SED slope $\alpha$")
    _style_axis(ax)
    return _save_figure(ctx, fig, "tier_b_sed_alpha_vs_halpha")


def plot_period_histogram(ctx: PaperFigureContext, dippers: pd.DataFrame | None = None) -> FigureResult:
    dippers = load_dippers(ctx.review_db) if dippers is None else dippers
    period = pd.to_numeric(dippers["period_consensus_days"], errors="coerce")
    period = period.loc[period.gt(0)]
    fig, ax = plt.subplots(figsize=(FIG_SINGLE_COL_WIDTH * 1.2, FIG_SINGLE_COL_WIDTH * 0.85), layout="constrained")
    bins = np.geomspace(max(period.min() * 0.8, 0.05), period.max() * 1.2, 24)
    ax.hist(period, bins=bins, color="#2166ac", edgecolor="black", alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlabel("Consensus period [days]")
    ax.set_ylabel("Count")
    _style_axis(ax)
    return _save_figure(ctx, fig, "tier_b_period_histogram")


def plot_halpha_vs_m(ctx: PaperFigureContext, dippers: pd.DataFrame | None = None) -> FigureResult:
    dippers = load_dippers(ctx.review_db) if dippers is None else dippers
    m = pd.to_numeric(dippers["stats_variability_flux_asymmetry_m"], errors="coerce")
    ha = pd.to_numeric(dippers["iphas_r_ha"].fillna(dippers["vphas_r_ha"]), errors="coerce")
    good = np.isfinite(m) & np.isfinite(ha)
    fig, ax = plt.subplots(figsize=(FIG_SINGLE_COL_WIDTH, FIG_SINGLE_COL_WIDTH * 0.85), layout="constrained")
    ax.scatter(m[good], ha[good], s=34, c="#d85c1c", edgecolors="black", linewidths=0.25)
    ax.set_xlabel(r"Flux asymmetry $M$")
    ax.set_ylabel(LABEL_R_HALPHA)
    _style_axis(ax)
    return _save_figure(ctx, fig, "tier_b_halpha_vs_m")


def plot_sed_alpha_vs_ir_rms(ctx: PaperFigureContext, dippers: pd.DataFrame | None = None) -> FigureResult:
    dippers = load_dippers(ctx.review_db) if dippers is None else dippers.copy()
    quality_path = ctx.sed_dir / "marked_dipper_allwise_quality.parquet"
    if quality_path.exists():
        quality = pd.read_parquet(quality_path)
        quality["candidate_id"] = quality["candidate_id"].astype(str)
        merge_cols = ["candidate_id"] + [
            col
            for col in (
                "allwise_mep_w1_intrinsic_scatter",
                "allwise_mep_w1_robust_sigma",
                "allwise_mep_w1_range",
            )
            if col in quality.columns
        ]
        if len(merge_cols) > 1:
            dippers = dippers.merge(quality[merge_cols], on="candidate_id", how="left", suffixes=("", "_quality"))

    alpha = _column_as_series(dippers, "sed_alpha")
    ir_rms = _column_as_series(dippers, "allwise_mep_w1_intrinsic_scatter")
    if not np.isfinite(ir_rms).any():
        ir_rms = _column_as_series(dippers, "allwise_mep_w1_robust_sigma")
    if not np.isfinite(ir_rms).any():
        ir_rms = _column_as_series(dippers, "allwise_mep_w1_range")
    if not np.isfinite(ir_rms).any():
        ir_rms = _column_as_series(dippers, "stats_intrinsic_sigma_mag")

    good = np.isfinite(alpha.to_numpy()) & np.isfinite(ir_rms.to_numpy())
    if not good.any():
        raise RuntimeError("No finite SED alpha and IR scatter measurements for dippers")

    fig, ax = plt.subplots(figsize=(FIG_SINGLE_COL_WIDTH, FIG_SINGLE_COL_WIDTH * 0.85), layout="constrained")
    ax.scatter(ir_rms.to_numpy()[good], alpha.to_numpy()[good], s=34, c="#7b3294", edgecolors="black", linewidths=0.25)
    ax.set_xlabel(r"AllWISE W1 variability [mag]")
    ax.set_ylabel(r"SED slope $\alpha$")
    _style_axis(ax)
    return _save_figure(ctx, fig, "tier_b_sed_alpha_vs_ir_rms")


def _galactic_radians(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gl = _column_as_series(frame, "gal_l").to_numpy(dtype=float)
    gb = _column_as_series(frame, "gal_b").to_numpy(dtype=float)
    ra = _column_as_series(frame, "ra").to_numpy(dtype=float)
    dec = _column_as_series(frame, "dec").to_numpy(dtype=float)
    need = (~np.isfinite(gl) | ~np.isfinite(gb)) & np.isfinite(ra) & np.isfinite(dec)
    if need.any():
        coords = SkyCoord(ra=ra[need] * u.deg, dec=dec[need] * u.deg)
        gl[need] = coords.galactic.l.deg
        gb[need] = coords.galactic.b.deg
    l_rad = np.deg2rad(((gl + 180.0) % 360.0) - 180.0)
    b_rad = np.deg2rad(gb)
    good = np.isfinite(l_rad) & np.isfinite(b_rad)
    return l_rad, b_rad, good


def plot_mollweide_sky_sfr(ctx: PaperFigureContext, dippers: pd.DataFrame | None = None) -> FigureResult:
    dippers = load_dippers(ctx.review_db) if dippers is None else dippers
    sfr_path = ctx.repo_root / "malca" / "data" / "star_forming_regions.csv"
    sfrs = pd.read_csv(sfr_path)

    fig = plt.figure(figsize=(10.0, 5.2), layout="constrained")
    ax = fig.add_subplot(111, projection="mollweide")
    ax.set_facecolor("0.92")
    ax.grid(True, color="0.75", lw=0.4)

    for _, row in sfrs.iterrows():
        l_rad = np.deg2rad((float(row["l_deg"]) + 180.0) % 360.0 - 180.0)
        b_rad = np.deg2rad(float(row["b_deg"]))
        radius = np.deg2rad(float(row["radius_deg"]))
        circle = patches.Circle(
            (l_rad, b_rad),
            radius,
            transform=ax.transData,
            fill=False,
            ec="0.35",
            lw=0.6,
            alpha=0.8,
        )
        ax.add_patch(circle)

    l, b, good = _galactic_radians(dippers)
    ax.scatter(l[good], b[good], s=16, c="#d85c1c", edgecolors="black", linewidths=0.2, alpha=0.85, zorder=4)

    pmra = pd.to_numeric(dippers.get("pmra"), errors="coerce")
    pmdec = pd.to_numeric(dippers.get("pmdec"), errors="coerce")
    parallax = pd.to_numeric(dippers.get("parallax"), errors="coerce").clip(lower=1e-4)
    pm_good = good & np.isfinite(pmra) & np.isfinite(pmdec) & np.isfinite(parallax)
    if pm_good.any():
        pm_coords = SkyCoord(
            ra=pd.to_numeric(dippers.loc[pm_good, "ra"], errors="coerce").to_numpy() * u.deg,
            dec=pd.to_numeric(dippers.loc[pm_good, "dec"], errors="coerce").to_numpy() * u.deg,
            pm_ra_cosdec=pmra[pm_good].to_numpy() * u.mas / u.yr,
            pm_dec=pmdec[pm_good].to_numpy() * u.mas / u.yr,
            distance=1000.0 / parallax[pm_good].to_numpy() * u.pc,
            frame="icrs",
        )
        pm_gal = pm_coords.galactic
        scale = 0.015
        ax.quiver(
            l[pm_good],
            b[pm_good],
            pm_gal.pm_l_cosb.value * scale,
            pm_gal.pm_b.value * scale,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.0025,
            color="#2166ac",
            alpha=0.7,
            zorder=5,
        )
    ax.set_title("Dippers, SFR footprints, and proper-motion vectors")
    return _save_figure(ctx, fig, "tier_b_mollweide_sky_sfr")


def plot_age_histogram(ctx: PaperFigureContext, dippers: pd.DataFrame | None = None) -> FigureResult:
    dippers = load_dippers(ctx.review_db) if dippers is None else dippers
    age_gyr = pd.to_numeric(dippers["age50"], errors="coerce")
    age_myr = age_gyr.loc[np.isfinite(age_gyr) & age_gyr.gt(0)] * 1e3
    fig, ax = plt.subplots(figsize=(FIG_SINGLE_COL_WIDTH, FIG_SINGLE_COL_WIDTH * 0.85), layout="constrained")
    if age_myr.empty:
        ax.text(0.5, 0.5, "No finite age50 values", transform=ax.transAxes, ha="center")
    else:
        ax.hist(age_myr, bins=20, color="#4393c3", edgecolor="black", alpha=0.8)
        ax.set_xlabel("Age [Myr]")
        ax.set_ylabel("Count")
    _style_axis(ax)
    return _save_figure(ctx, fig, "tier_b_age_histogram")


def plot_dip_depth_histogram(ctx: PaperFigureContext, dippers: pd.DataFrame | None = None) -> FigureResult:
    dippers = load_dippers(ctx.review_db) if dippers is None else dippers
    depth = 1.0 - np.power(10.0, -0.4 * pd.to_numeric(dippers["dip_best_mag_event"], errors="coerce"))
    depth = depth.loc[np.isfinite(depth) & depth.gt(0) & depth.lt(1)]
    fig, ax = plt.subplots(figsize=(FIG_SINGLE_COL_WIDTH, FIG_SINGLE_COL_WIDTH * 0.85), layout="constrained")
    ax.hist(depth, bins=20, color="#455a54", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Fractional dip depth")
    ax.set_ylabel("Count")
    _style_axis(ax)
    return _save_figure(ctx, fig, "tier_b_dip_depth_histogram")


def plot_dip_duration_histogram(ctx: PaperFigureContext, dippers: pd.DataFrame | None = None) -> FigureResult:
    dippers = load_dippers(ctx.review_db) if dippers is None else dippers
    duration = pd.to_numeric(dippers["dip_max_run_duration"], errors="coerce")
    duration = duration.loc[duration.gt(0)]
    fig, ax = plt.subplots(figsize=(FIG_SINGLE_COL_WIDTH * 1.1, FIG_SINGLE_COL_WIDTH * 0.85), layout="constrained")
    bins = np.geomspace(max(duration.min() * 0.8, 0.05), duration.max() * 1.2, 20)
    ax.hist(duration, bins=bins, color="#455a54", edgecolor="black", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("Dip duration [days]")
    ax.set_ylabel("Count")
    _style_axis(ax)
    return _save_figure(ctx, fig, "tier_b_dip_duration_histogram")


def run_existing_cmd_script(ctx: PaperFigureContext) -> FigureResult:
    script = ctx.repo_root / "scripts" / "plot_march18_review_cmd.py"
    subprocess.run(
        [sys.executable, str(script), "--preset", "july1-dippers"],
        cwd=ctx.repo_root,
        check=True,
    )
    out = ctx.run_root / "results" / "july1_review_cmd_dippers.png"
    return [out] if out.exists() else []


def generate_tier_a(ctx: PaperFigureContext, *, include_cmd_script: bool = False) -> dict[str, FigureResult]:
    apply_publication_rcparams(plt)
    plt.rcParams.update({**PUBLICATION_STYLE, "axes.formatter.use_mathtext": True})
    survey = load_survey_candidates(ctx.review_db)
    dippers = survey.loc[survey["is_dipper"]].copy()
    outputs: dict[str, FigureResult] = {}
    outputs["amplitude_vs_median_g"] = plot_amplitude_vs_median_g(ctx, survey)
    outputs["error_vs_median_g"] = plot_error_vs_median_g(ctx, survey)
    outputs["mq_diagram"] = plot_mq_diagram(ctx, survey)
    outputs["dip_depth_timescale"] = plot_dip_depth_timescale(ctx, dippers)
    outputs["ir_excess_vs_bprp"] = plot_ir_excess_vs_bprp(ctx, dippers)
    outputs["ks_w2_w4_dered"] = plot_ks_w2_w4_dered(ctx, dippers)
    outputs["extinction_corrected_cmd"] = plot_extinction_corrected_cmd(ctx, dippers)
    outputs["representative_seds"] = plot_representative_seds(ctx)
    outputs["sed_with_residuals"] = plot_sed_with_residuals(ctx)
    if include_cmd_script:
        cmd_paths = run_existing_cmd_script(ctx)
        if cmd_paths:
            outputs["cmd_script"] = (cmd_paths, None)
    if ctx.show_inline:
        for _name, (paths, fig) in outputs.items():
            if fig is not None:
                show_figure(paths, fig)
    return outputs


def generate_tier_b(ctx: PaperFigureContext) -> dict[str, FigureResult]:
    apply_publication_rcparams(plt)
    plt.rcParams.update({**PUBLICATION_STYLE, "axes.formatter.use_mathtext": True})
    dippers = load_dippers(ctx.review_db)
    outputs: dict[str, FigureResult] = {}
    outputs["ks_w3_w4_observed"] = plot_ks_w3_w4_observed(ctx, dippers)
    outputs["sed_alpha_vs_halpha"] = plot_sed_alpha_vs_halpha(ctx, dippers)
    outputs["period_histogram"] = plot_period_histogram(ctx, dippers)
    outputs["halpha_vs_m"] = plot_halpha_vs_m(ctx, dippers)
    outputs["sed_alpha_vs_ir_rms"] = plot_sed_alpha_vs_ir_rms(ctx, dippers)
    outputs["mollweide_sky_sfr"] = plot_mollweide_sky_sfr(ctx, dippers)
    outputs["age_histogram"] = plot_age_histogram(ctx, dippers)
    outputs["dip_depth_histogram"] = plot_dip_depth_histogram(ctx, dippers)
    outputs["dip_duration_histogram"] = plot_dip_duration_histogram(ctx, dippers)
    if ctx.show_inline:
        for _name, (paths, fig) in outputs.items():
            if fig is not None:
                show_figure(paths, fig)
    return outputs
