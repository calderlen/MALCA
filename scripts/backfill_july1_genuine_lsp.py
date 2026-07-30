"""Backfill genuine Lomb--Scargle measurements for the July 1 review run.

This intentionally reuses the completed PDM/CE solution and recomputes only
the fields whose historical ``lsp_*`` values were compatibility aliases.  The
worker uses the same light-curve preparation, block bootstrap, alias policy,
and deterministic seed as the production periodicity validator.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
import shutil
import zlib

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from malca.config import BAD_CAMERA_SCATTER_RATIO_THRESHOLD
from malca.core.phase import align_v_to_g_magnitude
from malca.core.stats import bootstrap_lomb_scargle
from malca.io.lightcurve_io import load_lightcurve_df, to_asassn_algorithm_frame
from malca.io.table_io import read_feature_table
from malca.review.store import db_connect, merge_candidate_results
from malca.stv.periodicity_gate import prepare_periodicity_lightcurve


DEFAULT_RUN_DIR = Path("output/runs/dat3-full-extended_2026-07-01-v4")
DEFAULT_SOURCE = DEFAULT_RUN_DIR / "results" / "lc_events_vetted_periodicity_100boot_multiples.parquet"
DEFAULT_CHECKPOINT = DEFAULT_RUN_DIR / "checkpoints" / "genuine_lsp_100boot" / "lsp_checkpoint.parquet"
DEFAULT_OUTPUT = DEFAULT_RUN_DIR / "results" / "genuine_lsp_100boot.parquet"
DEFAULT_REVIEW_DB = DEFAULT_RUN_DIR / "review" / "review.db"

LSP_BACKFILL_VERSION = "genuine_lsp_block_bootstrap_v1"
LSP_VALUE_COLUMNS = (
    "lsp_power",
    "lsp_period",
    "lsp_bootstrap_sig",
    "lsp_is_alias",
    "lsp_is_significant",
)


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _lsp_only_worker(task: tuple[str, str, str, int, float, bool]) -> dict[str, object]:
    candidate_id, original_path, resolved_path, n_bootstrap, significance_level, exclude_aliases = task
    path = Path(resolved_path)
    result: dict[str, object] = {
        "candidate_id": candidate_id,
        "lc_path": original_path,
        "resolved_path": resolved_path,
        "lsp_backfill_version": LSP_BACKFILL_VERSION,
        "lsp_n_bootstrap": int(n_bootstrap),
        "lsp_significance_level": float(significance_level),
        "lsp_exclude_aliases": bool(exclude_aliases),
    }
    try:
        input_stat = path.stat()
        result["lsp_input_size"] = int(input_stat.st_size)
        result["lsp_input_mtime_ns"] = int(input_stat.st_mtime_ns)

        lightcurve = load_lightcurve_df(
            path,
            filter_bad_cameras_enabled=True,
            bad_camera_scatter_ratio=float(BAD_CAMERA_SCATTER_RATIO_THRESHOLD),
        )
        lightcurve = to_asassn_algorithm_frame(lightcurve)
        lightcurve = prepare_periodicity_lightcurve(lightcurve)
        sort_columns = [column for column in ("v_g_band", "JD") if column in lightcurve.columns]
        if sort_columns:
            lightcurve = lightcurve.sort_values(sort_columns, kind="stable").reset_index(drop=True)
        lightcurve, _ = align_v_to_g_magnitude(lightcurve)

        seed = int(zlib.crc32(original_path.encode("utf-8")) & 0xFFFFFFFF)
        lsp = bootstrap_lomb_scargle(
            lightcurve["JD"].to_numpy(dtype=float),
            lightcurve["mag"].to_numpy(dtype=float),
            lightcurve["error"].to_numpy(dtype=float),
            n_bootstrap=int(n_bootstrap),
            exclude_alias_periods=bool(exclude_aliases),
            significance_level=float(significance_level),
            random_state=(seed + 2) & 0xFFFFFFFF,
        )
        result.update(
            {
                "lsp_power": lsp.get("ls_power", np.nan),
                "lsp_period": lsp.get("ls_period_days", np.nan),
                "lsp_bootstrap_sig": lsp.get("ls_bootstrap_sig", np.nan),
                "lsp_is_alias": bool(lsp.get("ls_is_alias", False)),
                "lsp_is_significant": bool(lsp.get("ls_is_significant", False)),
                "lsp_bootstrap_attempted": int(lsp.get("ls_bootstrap_attempted", 0)),
                "lsp_bootstrap_successful": int(lsp.get("ls_bootstrap_successful", 0)),
                "lsp_bootstrap_method": str(lsp.get("ls_bootstrap_method", "")),
                "lsp_status": str(lsp.get("ls_status", "")),
                "error": None,
            }
        )
    except Exception as exc:
        result.update(
            {
                "lsp_input_size": -1,
                "lsp_input_mtime_ns": -1,
                "lsp_power": np.nan,
                "lsp_period": np.nan,
                "lsp_bootstrap_sig": np.nan,
                "lsp_is_alias": False,
                "lsp_is_significant": False,
                "lsp_bootstrap_attempted": int(n_bootstrap),
                "lsp_bootstrap_successful": 0,
                "lsp_bootstrap_method": "observing_block_permutation",
                "lsp_status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return result


def _load_source_tasks(source: Path, run_dir: Path) -> pd.DataFrame:
    source_frame = read_feature_table(source, columns=("candidate_id", "lc_path"))
    source_frame = source_frame.drop_duplicates("candidate_id", keep="last").copy()
    source_frame["resolved_path"] = source_frame["lc_path"].map(
        lambda value: str(run_dir / "bundle_assets" / "lightcurves" / Path(str(value)).name)
    )
    missing = ~source_frame["resolved_path"].map(lambda value: Path(value).exists())
    if bool(missing.any()):
        examples = ", ".join(source_frame.loc[missing, "resolved_path"].head(5))
        raise FileNotFoundError(f"Missing {int(missing.sum())} bundled light curves; examples: {examples}")
    return source_frame


def _load_cached_results(
    checkpoint: Path,
    *,
    n_bootstrap: int,
    significance_level: float,
    exclude_aliases: bool,
) -> dict[str, dict[str, object]]:
    if not checkpoint.exists():
        return {}
    frame = pd.read_parquet(checkpoint)
    valid = (
        frame.get("lsp_backfill_version", pd.Series("", index=frame.index)).eq(LSP_BACKFILL_VERSION)
        & pd.to_numeric(frame.get("lsp_n_bootstrap"), errors="coerce").eq(int(n_bootstrap))
        & pd.to_numeric(frame.get("lsp_significance_level"), errors="coerce").eq(float(significance_level))
        & frame.get("lsp_exclude_aliases", pd.Series(False, index=frame.index)).astype(bool).eq(bool(exclude_aliases))
        & frame.get("error", pd.Series(None, index=frame.index)).isna()
    )
    return {
        str(row["candidate_id"]): row.to_dict()
        for _, row in frame.loc[valid].iterrows()
    }


def _checkpoint_frame(results: dict[str, dict[str, object]], source_order: dict[str, int]) -> pd.DataFrame:
    frame = pd.DataFrame(results.values())
    if frame.empty:
        return frame
    frame["_source_order"] = frame["candidate_id"].map(source_order)
    return frame.sort_values("_source_order", kind="stable").drop(columns="_source_order").reset_index(drop=True)


def _merge_into_review_db(frame: pd.DataFrame, review_db: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = review_db.with_name(f"{review_db.name}.pre_genuine_lsp_{stamp}.bak")
    shutil.copy2(review_db, backup)
    updates = frame.loc[frame["error"].isna(), ["candidate_id", *LSP_VALUE_COLUMNS]].copy()
    with db_connect(review_db) as conn:
        merge_candidate_results(conn, updates)
    return backup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-db", type=Path, default=DEFAULT_REVIEW_DB)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--n-bootstrap", type=int, default=100)
    parser.add_argument("--significance", type=float, default=0.01)
    parser.add_argument("--no-exclude-aliases", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--merge-review-db", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    review_db = args.review_db.expanduser().resolve()
    exclude_aliases = not bool(args.no_exclude_aliases)

    source_frame = _load_source_tasks(source, run_dir)
    source_order = {str(candidate_id): index for index, candidate_id in enumerate(source_frame["candidate_id"])}
    results = _load_cached_results(
        checkpoint,
        n_bootstrap=args.n_bootstrap,
        significance_level=args.significance,
        exclude_aliases=exclude_aliases,
    )
    pending = source_frame.loc[~source_frame["candidate_id"].astype(str).isin(results)].copy()
    tasks = [
        (
            str(row.candidate_id),
            str(row.lc_path),
            str(row.resolved_path),
            int(args.n_bootstrap),
            float(args.significance),
            bool(exclude_aliases),
        )
        for row in pending.itertuples(index=False)
    ]
    print(f"Loaded {len(results)} cached rows; computing {len(tasks)} of {len(source_frame)} candidates", flush=True)

    completed_since_checkpoint = 0
    if tasks:
        with Pool(processes=max(1, int(args.workers))) as pool:
            iterator = pool.imap_unordered(_lsp_only_worker, tasks, chunksize=1)
            for result in tqdm(iterator, total=len(tasks), unit="candidate"):
                results[str(result["candidate_id"])] = result
                completed_since_checkpoint += 1
                if completed_since_checkpoint >= max(1, int(args.checkpoint_interval)):
                    _atomic_write_parquet(_checkpoint_frame(results, source_order), checkpoint)
                    completed_since_checkpoint = 0

    final = _checkpoint_frame(results, source_order)
    _atomic_write_parquet(final, checkpoint)
    _atomic_write_parquet(final, output)
    errors = final["error"].notna() if "error" in final.columns else pd.Series(False, index=final.index)
    print(
        f"Wrote {len(final)} rows to {output}; errors={int(errors.sum())}; "
        f"finite_lsp={int(pd.to_numeric(final['lsp_period'], errors='coerce').notna().sum())}",
        flush=True,
    )
    if bool(errors.any()):
        raise RuntimeError(f"Genuine LSP backfill has {int(errors.sum())} worker errors; rerun to retry them")
    if len(final) != len(source_frame):
        raise RuntimeError(f"Genuine LSP backfill is incomplete: {len(final)} != {len(source_frame)}")

    if args.merge_review_db:
        backup = _merge_into_review_db(final, review_db)
        print(f"Merged genuine LSP fields into {review_db}; backup={backup}", flush=True)


if __name__ == "__main__":
    main()
