from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Mapping, Any
import time

import pandas as pd

from malca.nuclear.targets import normalize_nuclear_targets
from malca.nuclear.features import compute_lightcurve_feature_table
from malca.nuclear.redshift import resolve_redshift_spectral_types
from malca.nuclear.scoring import score_nuclear_candidates
from malca.nuclear.clagn_catalogs import load_known_clagn_catalogs, match_known_clagn_catalogs
from malca.run_context import init_pipeline_run_context, write_run_params, write_run_summary
from malca.table_io import write_parquet_table


@dataclass
class NuclearContextConfig:
    run_dir: Path = Path("output") / "runs" / "nuclear_context"
    cache_dir: Path | None = None
    checkpoint_dir: Path | None = None
    workers: int = 4
    chunk_size: int = 250
    show_progress: bool = False
    refresh_cache: bool = False
    skip_existing: bool = True
    fail_soft: bool = True

    run_lightcurve_features: bool = True
    run_characterize: bool = True
    run_ltv_crossmatch: bool = True
    run_vetting: bool = True
    run_external_lcs: bool = True
    run_spectra: bool = True
    run_host: bool = True
    run_radio: bool = True
    run_swift: bool = True
    run_redshift_spectra: bool = True
    run_clagn_catalogs: bool = True
    run_scores: bool = True

    atlas_token: str | None = None
    tns_api_key: str | None = None
    host_radius_arcsec: float = 5.0
    radio_radius_arcsec: float = 10.0
    swift_radius_arcsec: float = 10.0
    spectra_radius_arcsec: float = 3.0
    clagn_match_radius_arcsec: float = 3.0
    clagn_catalog_paths: Mapping[str, str | Path] | None = None

    extra_stage_kwargs: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def results_dir(self) -> Path:
        return Path(self.run_dir).expanduser() / "results"

    @property
    def review_dir(self) -> Path:
        return Path(self.run_dir).expanduser() / "review"

    @property
    def effective_cache_dir(self) -> Path:
        return Path(self.cache_dir or (Path(self.run_dir).expanduser() / "cache")).expanduser()

    @property
    def effective_checkpoint_dir(self) -> Path:
        return Path(self.checkpoint_dir or (Path(self.run_dir).expanduser() / "checkpoints")).expanduser()


def _stage_col(stage: str, suffix: str = "status") -> str:
    safe = stage.replace("-", "_")
    return f"nuclear_stage_{safe}_{suffix}"


def _mark_stage(df: pd.DataFrame, stage: str, status: str, error: str = "") -> pd.DataFrame:
    out = df.copy()
    out[_stage_col(stage)] = status
    if error:
        out[_stage_col(stage, "error")] = error
    elif _stage_col(stage, "error") not in out.columns:
        out[_stage_col(stage, "error")] = ""
    return out


def _merge_summary(df: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if summary is None or summary.empty or "candidate_id" not in summary.columns:
        return df
    base = df.copy()
    add = summary.copy()
    add["candidate_id"] = add["candidate_id"].astype(str)
    drop = [col for col in add.columns if col != "candidate_id" and col in base.columns]
    if drop:
        base = base.drop(columns=drop)
    return base.merge(add, on="candidate_id", how="left")


def _write_stage(df: pd.DataFrame, config: NuclearContextConfig, name: str) -> None:
    write_parquet_table(df, config.results_dir / f"nuclear_{name}.parquet")


def _run_dataframe_stage(
    df: pd.DataFrame,
    *,
    stage: str,
    config: NuclearContextConfig,
    func: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    t0 = time.perf_counter()
    try:
        out = func(df)
        out = _mark_stage(out, stage, "ok")
    except Exception as exc:
        if not config.fail_soft:
            raise
        out = _mark_stage(df, stage, "query_failed", str(exc))
    out[_stage_col(stage, "elapsed_s")] = round(time.perf_counter() - t0, 3)
    _write_stage(out, config, stage)
    return out


def _checkpoint(config: NuclearContextConfig, name: str) -> Path:
    path = config.effective_checkpoint_dir / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _cache(config: NuclearContextConfig, name: str) -> Path:
    path = config.effective_cache_dir / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run_nuclear_context(df: pd.DataFrame, config: NuclearContextConfig | None = None) -> pd.DataFrame:
    """Run the nuclear context stack and return the enriched/scored table."""
    config = config or NuclearContextConfig()
    ctx = init_pipeline_run_context("nuclear", Path(config.run_dir))
    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.review_dir.mkdir(parents=True, exist_ok=True)
    config.effective_cache_dir.mkdir(parents=True, exist_ok=True)
    config.effective_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_run_params(ctx, asdict(config))

    out = normalize_nuclear_targets(df)
    _write_stage(out, config, "targets")

    if config.run_lightcurve_features and "lc_path" in out.columns:
        def _features(frame: pd.DataFrame) -> pd.DataFrame:
            features = compute_lightcurve_feature_table(frame)
            return _merge_summary(frame, features)

        out = _run_dataframe_stage(out, stage="lightcurve_features", config=config, func=_features)

    if config.run_characterize:
        def _characterize(frame: pd.DataFrame) -> pd.DataFrame:
            from malca.characterize import characterize_candidates_df

            return characterize_candidates_df(
                frame,
                checkpoint_path=_checkpoint(config, "characterize"),
            )

        out = _run_dataframe_stage(out, stage="characterize", config=config, func=_characterize)

    if config.run_ltv_crossmatch:
        def _ltv_crossmatch(frame: pd.DataFrame) -> pd.DataFrame:
            from malca.ltv.crossmatch import crossmatch_all_catalogs

            return crossmatch_all_catalogs(
                frame,
                n_workers=config.workers,
                verbose=config.show_progress,
            )

        out = _run_dataframe_stage(out, stage="ltv_crossmatch", config=config, func=_ltv_crossmatch)

    if config.run_vetting:
        if not config.atlas_token:
            out = _mark_stage(out, "atlas", "not_configured", "atlas_token is not set")

        def _vet(frame: pd.DataFrame) -> pd.DataFrame:
            from malca.vetting import vet_candidates

            return vet_candidates(
                frame,
                run_atlas=bool(config.atlas_token),
                atlas_token=config.atlas_token,
                tns_api_key=config.tns_api_key,
                run_neowise_lc=True,
                atlas_output_dir=config.results_dir / "atlas",
                neowise_output_dir=config.results_dir / "neowise",
                checkpoint_path=_checkpoint(config, "vetting"),
                cache_dir=config.effective_cache_dir / "vetting",
                skip_existing=config.skip_existing,
                refresh_cache=config.refresh_cache,
            )

        out = _run_dataframe_stage(out, stage="vetting", config=config, func=_vet)

    if config.run_external_lcs:
        if not config.atlas_token:
            out = _mark_stage(out, "external_lcs_atlas", "not_configured", "atlas_token is not set")

        def _external(frame: pd.DataFrame) -> pd.DataFrame:
            from malca.vetting import fetch_external_lcs

            return fetch_external_lcs(
                frame,
                output_dir=config.results_dir / "external_lcs",
                run_atlas=bool(config.atlas_token),
                run_ztf=True,
                run_gaia_epoch=True,
                run_tess=True,
                run_neowise=True,
                run_ps1=True,
                run_crts=True,
                atlas_token=config.atlas_token,
                workers=config.workers,
                checkpoint_path=_checkpoint(config, "external_lcs"),
                refresh_cache=config.refresh_cache,
            )

        out = _run_dataframe_stage(out, stage="external_lcs", config=config, func=_external)

    if config.run_spectra:
        def _spectra(frame: pd.DataFrame) -> pd.DataFrame:
            from malca.enrich.spectra import run_spectra_availability

            _long, summary = run_spectra_availability(
                frame,
                out_dir=config.results_dir / "spectra",
                radius_arcsec=config.spectra_radius_arcsec,
                chunk_size=config.chunk_size,
                cache_file=_cache(config, "spectra_long"),
                checkpoint_path=_checkpoint(config, "spectra"),
                show_progress=config.show_progress,
            )
            return _merge_summary(frame, summary)

        out = _run_dataframe_stage(out, stage="spectra", config=config, func=_spectra)

    if config.run_host:
        def _host(frame: pd.DataFrame) -> pd.DataFrame:
            from malca.enrich.host import run_host_association

            _long, summary = run_host_association(
                frame,
                out_dir=config.results_dir / "host",
                radius_arcsec=config.host_radius_arcsec,
                chunk_size=config.chunk_size,
                cache_file=_cache(config, "host_long"),
                checkpoint_path=_checkpoint(config, "host"),
                show_progress=config.show_progress,
            )
            return _merge_summary(frame, summary)

        out = _run_dataframe_stage(out, stage="host", config=config, func=_host)

    if config.run_radio:
        def _radio(frame: pd.DataFrame) -> pd.DataFrame:
            from malca.enrich.radio import run_radio_enrichment

            _long, summary = run_radio_enrichment(
                frame,
                out_dir=config.results_dir / "radio",
                radius_arcsec=config.radio_radius_arcsec,
                chunk_size=config.chunk_size,
                cache_file=_cache(config, "radio_long"),
                checkpoint_path=_checkpoint(config, "radio"),
                show_progress=config.show_progress,
            )
            return _merge_summary(frame, summary)

        out = _run_dataframe_stage(out, stage="radio", config=config, func=_radio)

    if config.run_swift:
        def _swift(frame: pd.DataFrame) -> pd.DataFrame:
            from malca.enrich.swift import run_swift_enrichment

            _long, summary = run_swift_enrichment(
                frame,
                out_dir=config.results_dir / "swift",
                radius_arcsec=config.swift_radius_arcsec,
                chunk_size=config.chunk_size,
                cache_file=_cache(config, "swift_long"),
                checkpoint_path=_checkpoint(config, "swift"),
                show_progress=config.show_progress,
            )
            return _merge_summary(frame, summary)

        out = _run_dataframe_stage(out, stage="swift", config=config, func=_swift)

    if config.run_redshift_spectra:
        out = _run_dataframe_stage(out, stage="redshift_spectra", config=config, func=resolve_redshift_spectral_types)

    if config.run_clagn_catalogs:
        def _clagn(frame: pd.DataFrame) -> pd.DataFrame:
            catalog = load_known_clagn_catalogs(config.clagn_catalog_paths)
            return match_known_clagn_catalogs(
                frame,
                catalog,
                radius_arcsec=config.clagn_match_radius_arcsec,
            )

        out = _run_dataframe_stage(out, stage="clagn_catalogs", config=config, func=_clagn)

    if config.run_scores:
        out = _run_dataframe_stage(out, stage="scores", config=config, func=score_nuclear_candidates)

    write_parquet_table(out, config.results_dir / "nuclear_context.parquet")
    write_parquet_table(out, config.results_dir / "nuclear_scores.parquet")
    write_run_summary(
        ctx,
        {
            "n_candidates": int(len(out)),
            "output": str(config.results_dir / "nuclear_context.parquet"),
            "score_columns": [col for col in out.columns if col.endswith("_score")],
        },
    )
    return out
