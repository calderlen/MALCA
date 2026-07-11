#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord, search_around_sky

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from malca.nuclear.arbitration import arbitrate_nuclear_scores  # noqa: E402
from malca.nuclear.context import NuclearContextConfig, run_nuclear_context  # noqa: E402
from malca.plotting.lightcurve_publication import (  # noqa: E402
    FIG_TWO_COL_WIDTH,
    PUBLICATION_STYLE,
    _load_matplotlib,
    filter_lightcurve,
    load_lightcurve,
    plot_lightcurve,
    plot_lightcurve_panel,
    save_publication_figure,
    style_publication_axis,
)
from malca.io.table_io import write_feature_table  # noqa: E402


DEFAULT_TDE_CSV = Path("/Users/calder/Documents/scripts/gezari2021_table1_tde_asassn_coordinates.csv")
DEFAULT_ASASSN_INDEX = Path("input/asassn_index_masked_concat_cleaned_20250919_154524_brotli.parquet")
DEFAULT_LC_MANIFEST = Path("output/runs/dat3-full-extended_2026-07-01-v4/manifests/lc_manifest_all.parquet")
DEFAULT_RUN_DIR = Path("output/runs/nuclear/gezari2021_tde_asassn")
DEFAULT_INDEX_EXTRA_COLUMNS = (
    "refcat_id",
    "gaia_id",
    "hip_id",
    "tyc_id",
    "tmass_id",
    "allwise_id",
    "tic_id",
    "plx",
    "plx_d",
    "pm_ra",
    "pm_ra_d",
    "pm_dec",
    "pm_dec_d",
    "gaia_mag",
    "pstarrs_g_mag",
    "pstarrs_r_mag",
    "pstarrs_i_mag",
    "pstarrs_z_mag",
    "nstat",
)
SCORE_COLUMNS = (
    "agn_prior_score",
    "tde_candidate_score",
    "clagn_photometric_score",
)


class StrictPreflightError(RuntimeError):
    def __init__(self, message: str, diagnostics: pd.DataFrame) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class RecoveryPaths:
    run_dir: Path

    @property
    def results_dir(self) -> Path:
        return self.run_dir / "results"

    @property
    def plots_dir(self) -> Path:
        return self.run_dir / "plots"


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def slugify_name(value: object, *, prefix: str = "gezari2021") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return f"{prefix}_{text or 'unknown'}"


def _string_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().astype(str)


def _parquet_columns(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq

        return set(pq.read_schema(path).names)
    except Exception:
        return set(pd.read_parquet(path).columns)


def load_gezari_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "name",
        "survey",
        "waveband",
        "redshift",
        "log_lbb_erg_s",
        "log_tbb_k",
        "paper_reference",
        "ra_deg",
        "dec_deg",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"TDE CSV is missing required columns: {', '.join(missing)}")

    out = df.copy()
    out["candidate_id"] = out["name"].map(slugify_name)
    out["ra_deg"] = pd.to_numeric(out["ra_deg"], errors="coerce")
    out["dec_deg"] = pd.to_numeric(out["dec_deg"], errors="coerce")
    bad = out["ra_deg"].isna() | out["dec_deg"].isna()
    if bad.any():
        diagnostics = out.loc[bad, ["candidate_id", "name", "ra_deg", "dec_deg"]].copy()
        diagnostics["reason"] = "invalid_tde_coordinates"
        raise StrictPreflightError("TDE input contains invalid coordinates.", diagnostics)
    if out["candidate_id"].duplicated().any():
        dupes = out.loc[out["candidate_id"].duplicated(keep=False), ["candidate_id", "name"]].copy()
        dupes["reason"] = "duplicate_candidate_id"
        raise StrictPreflightError("TDE input contains duplicate candidate IDs.", dupes)
    return out


def load_asassn_index(path: Path, *, extra_columns: Sequence[str] = DEFAULT_INDEX_EXTRA_COLUMNS) -> pd.DataFrame:
    required = ["asas_sn_id", "ra_deg", "dec_deg"]
    available = _parquet_columns(path)
    missing = sorted(set(required) - available)
    if missing:
        raise ValueError(f"ASAS-SN index is missing required columns: {', '.join(missing)}")
    columns = [*required, *[col for col in extra_columns if col in available and col not in required]]
    df = pd.read_parquet(path, columns=columns)
    out = df.copy()
    out["asas_sn_id"] = _string_id(out["asas_sn_id"])
    out["ra_deg"] = pd.to_numeric(out["ra_deg"], errors="coerce")
    out["dec_deg"] = pd.to_numeric(out["dec_deg"], errors="coerce")
    return out.dropna(subset=["ra_deg", "dec_deg", "asas_sn_id"]).reset_index(drop=True)


def _nearest_diagnostics(
    tde: pd.DataFrame,
    index: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    by_target = pairs.groupby("target_pos") if not pairs.empty else {}
    for target_pos, target in tde.reset_index(drop=True).iterrows():
        matched = by_target.get_group(target_pos).sort_values("sep_arcsec") if not pairs.empty and target_pos in by_target.groups else pd.DataFrame()
        if matched.empty:
            nearest_id = ""
            nearest_sep = math.nan
            matched_ids = ""
            reason = "unmatched"
        else:
            nearest = matched.iloc[0]
            nearest_row = index.iloc[int(nearest["index_pos"])]
            nearest_id = nearest_row["asas_sn_id"]
            nearest_sep = float(nearest["sep_arcsec"])
            matched_ids = ";".join(index.iloc[matched["index_pos"].astype(int)]["asas_sn_id"].astype(str).tolist())
            reason = "ambiguous"
        rows.append(
            {
                "candidate_id": target["candidate_id"],
                "name": target["name"],
                "ra_deg": target["ra_deg"],
                "dec_deg": target["dec_deg"],
                "match_count": int(len(matched)),
                "reason": reason,
                "nearest_asas_sn_id": nearest_id,
                "nearest_sep_arcsec": nearest_sep,
                "matched_asas_sn_ids": matched_ids,
            }
        )
    return pd.DataFrame(rows)


def strict_crossmatch_asassn(
    tde: pd.DataFrame,
    index: pd.DataFrame,
    *,
    radius_arcsec: float = 5.0,
) -> pd.DataFrame:
    if tde.empty:
        return tde.copy()
    if index.empty:
        diagnostics = tde[["candidate_id", "name", "ra_deg", "dec_deg"]].copy()
        diagnostics["match_count"] = 0
        diagnostics["reason"] = "empty_asassn_index"
        raise StrictPreflightError("ASAS-SN index has no coordinate rows.", diagnostics)

    target_coords = SkyCoord(tde["ra_deg"].to_numpy(dtype=float) * u.deg, tde["dec_deg"].to_numpy(dtype=float) * u.deg)
    index_coords = SkyCoord(index["ra_deg"].to_numpy(dtype=float) * u.deg, index["dec_deg"].to_numpy(dtype=float) * u.deg)
    target_pos, index_pos, sep2d, _ = search_around_sky(target_coords, index_coords, float(radius_arcsec) * u.arcsec)
    pairs = pd.DataFrame(
        {
            "target_pos": target_pos.astype(int),
            "index_pos": index_pos.astype(int),
            "sep_arcsec": sep2d.arcsec.astype(float),
        }
    )

    counts = pairs.groupby("target_pos").size() if not pairs.empty else pd.Series(dtype=int)
    failed_positions = [
        pos
        for pos in range(len(tde))
        if int(counts.get(pos, 0)) != 1
    ]
    if failed_positions:
        diagnostics = _nearest_diagnostics(tde, index, pairs)
        diagnostics = diagnostics.loc[diagnostics["match_count"].ne(1)].reset_index(drop=True)
        raise StrictPreflightError(
            f"Expected exactly one ASAS-SN match within {radius_arcsec:g} arcsec for every TDE.",
            diagnostics,
        )

    one = pairs.sort_values(["target_pos", "sep_arcsec"]).drop_duplicates("target_pos", keep="first")
    tde_reset = tde.reset_index(drop=True).copy()
    matched_rows = index.iloc[one.sort_values("target_pos")["index_pos"].astype(int).to_numpy()].reset_index(drop=True).copy()
    matched_rows = matched_rows.rename(columns={"ra_deg": "asassn_ra_deg", "dec_deg": "asassn_dec_deg"})
    matched_rows["asassn_sep_arcsec"] = one.sort_values("target_pos")["sep_arcsec"].to_numpy(dtype=float)
    return pd.concat([tde_reset.reset_index(drop=True), matched_rows.reset_index(drop=True)], axis=1)


def _manifest_columns(manifest: pd.DataFrame) -> tuple[str, str]:
    id_col = "asas_sn_id" if "asas_sn_id" in manifest.columns else "source_id" if "source_id" in manifest.columns else None
    path_col = "lc_path" if "lc_path" in manifest.columns else "dat_path" if "dat_path" in manifest.columns else None
    if id_col is None:
        raise ValueError("LC manifest must contain 'asas_sn_id' or 'source_id'.")
    if path_col is None:
        raise ValueError("LC manifest must contain 'lc_path' or 'dat_path'.")
    return id_col, path_col


def _is_manifest_true(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value == 1)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def attach_lightcurve_manifest(matches: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    id_col, path_col = _manifest_columns(manifest)
    work = manifest.copy()
    work["_asassn_join_id"] = _string_id(work[id_col])
    matched_ids = set(_string_id(matches["asas_sn_id"]).tolist())
    matched_manifest = work.loc[work["_asassn_join_id"].isin(matched_ids)].copy()
    duplicate_ids = sorted(
        matched_manifest.loc[matched_manifest["_asassn_join_id"].duplicated(keep=False), "_asassn_join_id"]
        .unique()
        .tolist()
    )
    if duplicate_ids:
        diagnostics = pd.DataFrame({"asas_sn_id": duplicate_ids, "reason": "duplicate_manifest_rows"})
        raise StrictPreflightError("LC manifest has duplicate rows for matched IDs.", diagnostics)

    keep_cols = ["_asassn_join_id", path_col]
    for col in ("dat_exists", "mag_bin", "index_num", "index_csv", "lc_dir", "lc_dir_exists"):
        if col in work.columns and col not in keep_cols:
            keep_cols.append(col)
    right = work[keep_cols].copy()
    right = right.rename(columns={path_col: "lc_path"})

    out = matches.copy()
    out["_asassn_join_id"] = _string_id(out["asas_sn_id"])
    out = out.merge(right, on="_asassn_join_id", how="left", validate="many_to_one")
    out = out.rename(
        columns={
            "mag_bin": "asassn_mag_bin",
            "index_num": "asassn_index_num",
            "index_csv": "asassn_index_csv",
            "lc_dir": "asassn_lc_dir",
            "lc_dir_exists": "asassn_lc_dir_exists",
        }
    )
    out["asassn_manifest_source_id"] = out["_asassn_join_id"]

    diagnostics: list[dict[str, object]] = []
    for _, row in out.iterrows():
        reason = ""
        lc_value = row.get("lc_path", pd.NA)
        lc_path = "" if pd.isna(lc_value) else str(lc_value).strip()
        if not lc_path:
            reason = "missing_manifest_row"
        elif "dat_exists" not in out.columns or not _is_manifest_true(row.get("dat_exists")):
            reason = "dat_exists_not_true"
        elif not Path(lc_path).expanduser().exists():
            reason = "lc_path_missing_on_disk"
        if reason:
            diagnostics.append(
                {
                    "candidate_id": row.get("candidate_id", ""),
                    "name": row.get("name", ""),
                    "asas_sn_id": row.get("asas_sn_id", ""),
                    "lc_path": lc_path,
                    "dat_exists": row.get("dat_exists", pd.NA),
                    "reason": reason,
                }
            )
    if diagnostics:
        raise StrictPreflightError("Matched ASAS-SN sources are missing usable light curves.", pd.DataFrame(diagnostics))

    return out.drop(columns=["_asassn_join_id"]).reset_index(drop=True)


def write_preflight_artifacts(matches: pd.DataFrame, paths: RecoveryPaths) -> None:
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = paths.results_dir / "gezari2021_asassn_matches.csv"
    parquet_path = paths.results_dir / "gezari2021_asassn_matches.parquet"
    matches.to_csv(csv_path, index=False)
    matches.to_parquet(parquet_path, index=False)


def write_failure_artifact(error: StrictPreflightError, paths: RecoveryPaths) -> None:
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    error.diagnostics.to_csv(paths.results_dir / "gezari2021_unmatched_or_ambiguous.csv", index=False)


def build_recovery_summary(arbitrated: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "name",
        "candidate_id",
        "asas_sn_id",
        "asassn_sep_arcsec",
        "survey",
        "waveband",
        "redshift",
        "lc_path",
        "lc_feature_status",
        "nuc_n_points",
        "nuc_time_span_days",
        "agn_prior_score",
        "tde_candidate_score",
        "clagn_photometric_score",
        "nuclear_primary_hypothesis",
        "nuclear_primary_score",
        "nuclear_runner_up_hypothesis",
        "nuclear_runner_up_score",
        "nuclear_hypothesis_margin",
        "nuclear_hypothesis_status",
        "tde_candidate_reasons",
        "agn_prior_reasons",
        "clagn_reasons",
    ]
    summary = pd.DataFrame(index=arbitrated.index)
    for col in columns:
        summary[col] = arbitrated[col] if col in arbitrated.columns else pd.NA
    score = pd.to_numeric(summary["tde_candidate_score"], errors="coerce").fillna(-1.0)
    return summary.assign(_sort_score=score).sort_values("_sort_score", ascending=False).drop(columns=["_sort_score"])


def _plot_title(row: pd.Series) -> str:
    score = row.get("tde_candidate_score", np.nan)
    score_text = f"{float(score):.2f}" if pd.notna(score) else "NA"
    hypothesis = str(row.get("nuclear_primary_hypothesis", "unknown"))
    return f"{row.get('name', '')} | {row.get('asas_sn_id', '')} | TDE {score_text} | {hypothesis}"


def write_individual_lightcurve_plots(rows: pd.DataFrame, paths: RecoveryPaths) -> pd.DataFrame:
    out_dir = paths.plots_dir / "individual"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        candidate_id = str(row["candidate_id"])
        png = out_dir / f"{candidate_id}.png"
        pdf = out_dir / f"{candidate_id}.pdf"
        record = {
            "candidate_id": candidate_id,
            "name": row.get("name", ""),
            "plot_png": str(png),
            "plot_pdf": str(pdf),
            "plot_status": "ok",
            "plot_error": "",
        }
        try:
            lc = load_lightcurve(row["lc_path"])
            plot_df = filter_lightcurve(lc)
            if plot_df.empty:
                raise ValueError("No finite light-curve points remain after filtering.")
            for output in (png, pdf):
                plot_lightcurve(
                    lc,
                    plot_df,
                    output=output,
                    close=True,
                    title=_plot_title(row),
                    group_by="band",
                    legend="auto",
                    marker_size=3.0,
                )
        except Exception as exc:
            record["plot_status"] = "error"
            record["plot_error"] = str(exc)
        records.append(record)
    return pd.DataFrame(records)


def write_tde_score_grid(rows: pd.DataFrame, paths: RecoveryPaths) -> None:
    paths.plots_dir.mkdir(parents=True, exist_ok=True)
    plot = rows.copy()
    plot["tde_candidate_score"] = pd.to_numeric(plot.get("tde_candidate_score"), errors="coerce").fillna(0.0)
    plot = plot.sort_values("tde_candidate_score", ascending=True)
    height = max(6.0, 0.22 * len(plot) + 1.6)
    plt, _ = _load_matplotlib()
    with plt.rc_context(PUBLICATION_STYLE):
        fig, ax = plt.subplots(figsize=(FIG_TWO_COL_WIDTH, height))
        y = np.arange(len(plot))
        ax.barh(y, plot["tde_candidate_score"], color="#238b45", alpha=0.82, label="TDE")
        for col, color, label, offset in (
            ("agn_prior_score", "#756bb1", "AGN", 0.0),
            ("clagn_photometric_score", "#d95f0e", "CLAGN", 0.0),
        ):
            values = pd.to_numeric(plot[col], errors="coerce").fillna(0.0) if col in plot.columns else pd.Series(0.0, index=plot.index)
            ax.scatter(values, y + offset, s=11, color=color, alpha=0.75, label=label, zorder=3)
        labels = plot["name"].astype(str) + "  (" + plot["nuclear_primary_hypothesis"].fillna("unknown").astype(str) + ")"
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Score")
        ax.set_title("Gezari 2021 TDE candidates: nuclear scores")
        ax.legend(loc="lower right", frameon=False, ncol=3)
        style_publication_axis(ax)
        for suffix in ("png", "pdf"):
            save_publication_figure(fig, paths.plots_dir / f"gezari2021_tde_score_grid.{suffix}", close=False)
        plt.close(fig)


def _write_empty_pickups_grid(paths: RecoveryPaths) -> None:
    plt, _ = _load_matplotlib()
    with plt.rc_context(PUBLICATION_STYLE):
        fig, ax = plt.subplots(figsize=(FIG_TWO_COL_WIDTH, 2.5))
        ax.axis("off")
        ax.text(0.5, 0.5, "No TDE pickups at threshold.", ha="center", va="center", transform=ax.transAxes)
        for suffix in ("png", "pdf"):
            save_publication_figure(fig, paths.plots_dir / f"gezari2021_pickups_grid.{suffix}", close=False)
        plt.close(fig)


def write_pickups_grid(rows: pd.DataFrame, paths: RecoveryPaths, *, threshold: float = 0.5) -> None:
    paths.plots_dir.mkdir(parents=True, exist_ok=True)
    score = pd.to_numeric(rows.get("tde_candidate_score"), errors="coerce").fillna(0.0)
    primary = rows.get("nuclear_primary_hypothesis", pd.Series("", index=rows.index)).fillna("").astype(str)
    pickups = rows.loc[(primary == "tde") | (score >= threshold)].copy()
    if pickups.empty:
        _write_empty_pickups_grid(paths)
        return

    pickups["tde_candidate_score"] = pd.to_numeric(pickups["tde_candidate_score"], errors="coerce").fillna(0.0)
    pickups = pickups.sort_values("tde_candidate_score", ascending=False)
    ncols = 2
    nrows = int(math.ceil(len(pickups) / ncols))
    plt, _ = _load_matplotlib()
    with plt.rc_context(PUBLICATION_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_TWO_COL_WIDTH, max(2.4, 2.35 * nrows)), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (_, row) in zip(axes.ravel(), pickups.iterrows()):
            ax.axis("on")
            try:
                lc = load_lightcurve(row["lc_path"])
                plot_df = filter_lightcurve(lc)
                plot_lightcurve_panel(
                    ax,
                    lc,
                    plot_df,
                    title=_plot_title(row),
                    group_by="band",
                    legend="none",
                    marker_size=2.2,
                )
            except Exception as exc:
                ax.axis("off")
                ax.text(0.5, 0.5, f"{row.get('name', '')}\nplot failed: {exc}", ha="center", va="center", transform=ax.transAxes)
        for suffix in ("png", "pdf"):
            save_publication_figure(fig, paths.plots_dir / f"gezari2021_pickups_grid.{suffix}", close=False)
        plt.close(fig)


def run_recovery(
    *,
    tde_csv: Path,
    asassn_index: Path,
    lc_manifest: Path,
    run_dir: Path,
    match_radius_arcsec: float = 5.0,
    workers: int = 4,
    chunk_size: int = 56,
    atlas_token: str | None = None,
    tns_api_key: str | None = None,
    dry_run: bool = False,
    no_plots: bool = False,
    show_progress: bool = False,
) -> pd.DataFrame:
    paths = RecoveryPaths(run_dir=run_dir)
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.plots_dir.mkdir(parents=True, exist_ok=True)

    tde = load_gezari_table(tde_csv)
    index = load_asassn_index(asassn_index)
    matches = strict_crossmatch_asassn(tde, index, radius_arcsec=match_radius_arcsec)
    manifest = pd.read_parquet(lc_manifest)
    targets = attach_lightcurve_manifest(matches, manifest)
    write_preflight_artifacts(targets, paths)

    if dry_run:
        return targets

    config = NuclearContextConfig(
        run_dir=run_dir,
        workers=workers,
        chunk_size=chunk_size,
        atlas_token=atlas_token,
        tns_api_key=tns_api_key,
        show_progress=show_progress,
    )
    scored = run_nuclear_context(targets, config)
    arbitrated = arbitrate_nuclear_scores(scored)
    write_feature_table(arbitrated, paths.results_dir / "gezari2021_nuclear_arbitrated.parquet")
    summary = build_recovery_summary(arbitrated)
    summary.to_csv(paths.results_dir / "gezari2021_nuclear_recovery_summary.csv", index=False)

    if not no_plots:
        plot_status = write_individual_lightcurve_plots(arbitrated, paths)
        plot_status.to_csv(paths.results_dir / "gezari2021_plot_status.csv", index=False)
        write_tde_score_grid(arbitrated, paths)
        write_pickups_grid(arbitrated, paths)
    return arbitrated


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strictly crossmatch Gezari 2021 Table 1 TDEs to local ASAS-SN light curves and run MALCA nuclear context."
    )
    parser.add_argument("--tde-csv", type=Path, default=DEFAULT_TDE_CSV)
    parser.add_argument("--asassn-index", type=Path, default=DEFAULT_ASASSN_INDEX)
    parser.add_argument("--lc-manifest", type=Path, default=DEFAULT_LC_MANIFEST)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--match-radius-arcsec", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=56)
    parser.add_argument("--atlas-token", default=None)
    parser.add_argument("--tns-api-key", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Write match artifacts and stop before nuclear context.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG/PDF plot generation.")
    parser.add_argument("--show-progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_dir = resolve_repo_path(args.run_dir)
    paths = RecoveryPaths(run_dir=run_dir)
    try:
        out = run_recovery(
            tde_csv=resolve_repo_path(args.tde_csv),
            asassn_index=resolve_repo_path(args.asassn_index),
            lc_manifest=resolve_repo_path(args.lc_manifest),
            run_dir=run_dir,
            match_radius_arcsec=args.match_radius_arcsec,
            workers=args.workers,
            chunk_size=args.chunk_size,
            atlas_token=args.atlas_token,
            tns_api_key=args.tns_api_key,
            dry_run=args.dry_run,
            no_plots=args.no_plots,
            show_progress=args.show_progress,
        )
    except StrictPreflightError as exc:
        write_failure_artifact(exc, paths)
        print(f"Preflight failed: {exc}", file=sys.stderr)
        print(f"Wrote diagnostics to {paths.results_dir / 'gezari2021_unmatched_or_ambiguous.csv'}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"Dry run matched {len(out)} TDEs. Wrote {paths.results_dir / 'gezari2021_asassn_matches.csv'}")
    else:
        print(f"Nuclear recovery complete for {len(out)} TDEs.")
        print(f"Wrote results under {paths.results_dir}")
        if not args.no_plots:
            print(f"Wrote plots under {paths.plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
