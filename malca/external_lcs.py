"""CLI for fetching external light-curve products for candidate tables."""

from __future__ import annotations

import argparse
import os
from contextlib import closing
from pathlib import Path

import pandas as pd

from malca.candidates import select_passing_candidates_if_present
from malca.review.store import db_connect, merge_candidate_results
from malca.table_io import read_parquet_table, write_parquet_table


EXTERNAL_LC_PATTERNS = (
    "atlas_lc_*.parquet",
    "ztf_lc_*.parquet",
    "gaia_epoch_lc_*.parquet",
    "tess_lc_*.parquet",
    "neowise_lc_*.parquet",
    "ps1_lc_*.parquet",
    "crts_lc_*.parquet",
)

EXTERNAL_LC_COLUMNS = (
    "atlas_has_phot",
    "atlas_n_det_cyan",
    "atlas_n_det_orange",
    "atlas_cyan_range",
    "atlas_orange_range",
    "ztf_lc_n_det",
    "ztf_lc_g_range",
    "ztf_lc_r_range",
    "gaia_epoch_lc_n_g",
    "gaia_epoch_lc_g_range",
    "tess_n_sectors",
    "tess_total_points",
    "tess_flux_range",
    "neowise_n_epochs",
    "neowise_w1_range",
    "neowise_w2_range",
    "ps1_lc_n_points",
    "crts_lc_n_points",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="malca external-lcs",
        description="Fetch external light-curve products for MALCA candidates.",
    )
    parser.add_argument("input", type=Path, help="Input candidate Parquet file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path (default: <input>_external_lcs.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for per-candidate LC parquet files (default: input parent)",
    )
    parser.add_argument(
        "--review-db",
        type=Path,
        default=None,
        help="Optional review SQLite DB to merge enriched candidate fields into",
    )
    parser.add_argument(
        "--all-candidates",
        action="store_true",
        help="Fetch external light curves for all input rows instead of only failed_any=False passers.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (default: <output-dir>/<input>_external_lcs_CHECKPOINT.parquet)",
    )
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpoint resume/save")
    parser.add_argument("--refresh-cache", action="store_true", help="Ignore cached external LC files/status rows")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for supported fetchers")

    parser.add_argument(
        "--atlas",
        dest="run_atlas",
        action="store_true",
        default=False,
        help="Enable ATLAS forced photometry (default: disabled because it can poll slowly)",
    )
    parser.add_argument(
        "--no-atlas",
        dest="run_atlas",
        action="store_false",
        help="Disable ATLAS forced photometry (default)",
    )
    parser.add_argument(
        "--atlas-token",
        type=str,
        default=None,
        help="ATLAS forced-photometry token, or set MALCA_ATLAS_TOKEN/ATLAS_API_TOKEN",
    )
    parser.add_argument("--no-ztf", action="store_true", help="Skip ZTF light curves")
    parser.add_argument("--no-gaia-epoch", action="store_true", help="Skip Gaia epoch light curves")
    parser.add_argument("--no-tess", action="store_true", help="Skip TESS light curves")
    parser.add_argument("--no-neowise", action="store_true", help="Skip NEOWISE light curves")
    parser.add_argument("--no-ps1", action="store_true", help="Skip Pan-STARRS light curves")
    parser.add_argument("--no-crts", action="store_true", help="Skip CRTS light curves")
    return parser


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_external_lcs.parquet")


def _default_checkpoint_path(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}_external_lcs_CHECKPOINT.parquet"


def _ensure_candidate_id(df: pd.DataFrame) -> pd.DataFrame:
    if "candidate_id" in df.columns:
        return df
    if "asas_sn_id" not in df.columns:
        return df
    df = df.copy()
    df["candidate_id"] = df["asas_sn_id"].astype(str)
    return df


def _print_output_counts(output_dir: Path) -> None:
    print("\nExternal LC files:")
    for pattern in EXTERNAL_LC_PATTERNS:
        print(f"  {pattern}: {sum(1 for _ in output_dir.glob(pattern))}")


def _merge_frame(out: pd.DataFrame) -> pd.DataFrame:
    id_cols = [c for c in ("candidate_id", "asas_sn_id") if c in out.columns]
    value_cols = [c for c in EXTERNAL_LC_COLUMNS if c in out.columns]
    return out[id_cols + value_cols].copy()


def run(args: argparse.Namespace) -> Path:
    input_path = args.input.expanduser()
    output_path = (args.output or _default_output_path(input_path)).expanduser()
    output_dir = (args.output_dir or input_path.parent).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = None
    if not args.no_checkpoint:
        checkpoint_path = (args.checkpoint or _default_checkpoint_path(input_path, output_dir)).expanduser()

    df = read_parquet_table(input_path)
    df = _ensure_candidate_id(df)
    if not args.all_candidates:
        df = select_passing_candidates_if_present(df, printer=print)
    print(f"Loaded {len(df)} candidates from {input_path}")
    print(f"Writing per-candidate LC files to {output_dir}")

    from malca.vetting import fetch_external_lcs

    out = fetch_external_lcs(
        df,
        output_dir=output_dir,
        run_atlas=args.run_atlas,
        run_ztf=not args.no_ztf,
        run_gaia_epoch=not args.no_gaia_epoch,
        run_tess=not args.no_tess,
        run_neowise=not args.no_neowise,
        run_kepler=False,
        run_aavso=False,
        run_ps1=not args.no_ps1,
        run_crts=not args.no_crts,
        atlas_token=args.atlas_token or os.environ.get("MALCA_ATLAS_TOKEN") or os.environ.get("ATLAS_API_TOKEN"),
        workers=args.workers,
        checkpoint_path=checkpoint_path,
        refresh_cache=args.refresh_cache,
    )

    write_parquet_table(out, output_path)
    print(f"\nSaved external-LC table to {output_path}")

    if args.review_db:
        review_db = args.review_db.expanduser()
        merge_df = _merge_frame(out)
        with closing(db_connect(review_db)) as conn:
            updated = merge_candidate_results(conn, merge_df)
        print(f"Merged external-LC fields into {review_db} ({updated} candidates updated)")

    if checkpoint_path and checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Checkpoint removed: {checkpoint_path}")

    _print_output_counts(output_dir)
    return output_path


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
