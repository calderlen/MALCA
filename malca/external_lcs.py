"""CLI for fetching external light-curve products for candidate tables."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import pandas as pd

from malca.candidates import select_passing_candidates_if_present
from malca.feature_layers import feature_mapping_get, with_feature_columns
from malca.review.store import db_connect, merge_candidate_results
from malca.table_io import read_feature_table, write_feature_table

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None


EXTERNAL_LC_PATTERNS = (
    "atlas_lc_*.parquet",
    "ztf_lc_*.parquet",
    "gaia_epoch_lc_*.parquet",
    "tess_lc_*.parquet",
    "neowise_lc_*.parquet",
    "kepler_lc_*.parquet",
    "aavso_lc_*.parquet",
    "ogle_lc_*.parquet",
    "stripe82_lc_*.parquet",
    "allwise_mep_lc_*.parquet",
    "vvvx_virac_lc_*.parquet",
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
    "kepler_n_quarters",
    "kepler_total_points",
    "kepler_flux_range",
    "aavso_lc_n_points",
    "ogle_lc_n_points",
    "ogle_lc_i_range",
    "ogle_lc_v_range",
    "stripe82_lc_n_points",
    "stripe82_lc_u_range",
    "stripe82_lc_g_range",
    "stripe82_lc_r_range",
    "stripe82_lc_i_range",
    "stripe82_lc_z_range",
    "allwise_mep_n_epochs",
    "allwise_mep_w1_range",
    "allwise_mep_w2_range",
    "allwise_mep_w3_range",
    "allwise_mep_w4_range",
    "vvvx_virac_n_epochs",
    "vvvx_virac_z_range",
    "vvvx_virac_y_range",
    "vvvx_virac_j_range",
    "vvvx_virac_h_range",
    "vvvx_virac_ks_range",
    "ps1_lc_n_points",
    "crts_lc_n_points",
)

EXTERNAL_LC_INPUT_COLUMNS = (
    "ra",
    "dec",
    "ra_deg",
    "dec_deg",
    "gaia_id",
    "source_id",
    "gaia_epoch_available",
    "gaia_epoch_n_obs",
    "gaia_epoch_g_range",
    "period_ogle_name",
    "period_ogle_match",
    "period_ogle_days",
    "simbad_main_id",
    "asassn_var_name",
    "vsx_name",
    "tns_name",
    "ztf_var_name",
    "jd_first",
    "jd_last",
    "stats_jd_start",
    "stats_jd_end",
    "failed_any",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="malca external-lcs",
        description="Fetch external light-curve products for MALCA candidates.",
    )
    parser.add_argument("input", type=Path, help="Input candidate Parquet file or review SQLite DB")
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
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only rebuild summary fields from existing per-candidate LC files/status rows; do not query remote services.",
    )
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
    parser.add_argument("--no-kepler", action="store_true", help="Skip Kepler/K2 light curves")
    parser.add_argument("--no-aavso", action="store_true", help="Skip AAVSO light curves")
    parser.add_argument("--no-ogle", action="store_true", help="Skip OGLE OCVS light curves")
    parser.add_argument("--no-stripe82", action="store_true", help="Skip SDSS Stripe 82 light curves")
    parser.add_argument("--no-allwise-mep", action="store_true", help="Skip AllWISE Multiepoch light curves")
    parser.add_argument("--no-vvvx-virac", action="store_true", help="Skip VVVX/VIRAC2 light curves")
    parser.add_argument("--no-ps1", action="store_true", help="Skip Pan-STARRS light curves")
    parser.add_argument("--no-crts", action="store_true", help="Skip CRTS light curves")
    return parser


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_external_lcs.parquet")


def _default_checkpoint_path(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}_external_lcs_CHECKPOINT.parquet"


def _looks_like_review_db(path: Path) -> bool:
    return path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}


def _json_mapping(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if value in (None, "", b""):
        return {}
    try:
        parsed = json.loads(value) if isinstance(value, str) else {}
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _missing_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() in {"", "[]", "{}", "nan", "None", "null"}:
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    if isinstance(missing, bool):
        return missing
    try:
        return bool(missing.all())
    except Exception:
        pass
    return False


def _review_payload_value(payload: dict, column: str) -> object:
    value = feature_mapping_get(payload, column, None)
    if not _missing_value(value):
        return value
    nested = _json_mapping(payload.get("payload_json"))
    value = feature_mapping_get(nested, column, None)
    if not _missing_value(value):
        return value
    if column == "ra":
        value = feature_mapping_get(nested, "ra_deg", None)
    elif column == "dec":
        value = feature_mapping_get(nested, "dec_deg", None)
    elif column == "ra_deg":
        value = feature_mapping_get(nested, "ra", None)
    elif column == "dec_deg":
        value = feature_mapping_get(nested, "dec", None)
    return value


def _hydrate_review_db_input(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    if "payload_json" not in df.columns:
        return df
    out = df.copy()
    payloads = [_json_mapping(raw) for raw in out["payload_json"]]
    for column in columns:
        values = pd.Series([_review_payload_value(payload, column) for payload in payloads], index=out.index)
        has_value = ~values.map(_missing_value)
        if not has_value.any():
            continue
        if column not in out.columns:
            out[column] = values
            continue
        missing = out[column].map(_missing_value)
        out.loc[missing & has_value, column] = values.loc[missing & has_value]
    for target, fallback in (("ra", "ra_deg"), ("dec", "dec_deg"), ("ra_deg", "ra"), ("dec_deg", "dec")):
        if fallback not in out.columns:
            continue
        values = pd.to_numeric(out[fallback], errors="coerce")
        has_value = values.notna()
        if target not in out.columns:
            out[target] = values
            continue
        missing = out[target].map(_missing_value)
        out.loc[missing & has_value, target] = values.loc[missing & has_value]
    return out


def _read_input_candidates(input_path: Path) -> pd.DataFrame:
    if not _looks_like_review_db(input_path):
        return read_feature_table(input_path)
    with closing(db_connect(input_path)) as conn:
        df = pd.read_sql_query("SELECT * FROM candidates", conn)
    return _hydrate_review_db_input(df, EXTERNAL_LC_INPUT_COLUMNS)


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


def _source_run_flags(args: argparse.Namespace) -> dict[str, bool]:
    return {
        "atlas": bool(args.run_atlas),
        "ztf": not args.no_ztf,
        "gaia_epoch": not args.no_gaia_epoch,
        "tess": not args.no_tess,
        "neowise": not args.no_neowise,
        "kepler": not args.no_kepler,
        "aavso": not args.no_aavso,
        "ogle": not args.no_ogle,
        "stripe82": not args.no_stripe82,
        "allwise_mep": not args.no_allwise_mep,
        "vvvx_virac": not args.no_vvvx_virac,
        "ps1": not args.no_ps1,
        "crts": not args.no_crts,
    }


def _cache_only_specs():
    from malca import vetting

    return {
        "atlas": {
            "module": "ATLAS LCs",
            "prefix": "atlas_lc",
            "summary_cols": ["atlas_has_phot", "atlas_n_det_cyan", "atlas_n_det_orange", "atlas_cyan_range", "atlas_orange_range"],
            "match_col": "atlas_has_phot",
            "summarize": vetting._summarize_atlas_lc,
        },
        "ztf": {
            "module": "ZTF LCs",
            "prefix": "ztf_lc",
            "summary_cols": ["ztf_lc_n_det", "ztf_lc_g_range", "ztf_lc_r_range"],
            "match_col": "ztf_lc_n_det",
            "summarize": vetting._summarize_ztf_lc,
        },
        "gaia_epoch": {
            "module": "Gaia epoch LCs",
            "prefix": "gaia_epoch_lc",
            "summary_cols": ["gaia_epoch_lc_n_g", "gaia_epoch_lc_g_range"],
            "match_col": "gaia_epoch_lc_n_g",
            "summarize": vetting._summarize_gaia_epoch_lc,
        },
        "tess": {
            "module": "TESS LCs",
            "prefix": "tess_lc",
            "summary_cols": ["tess_n_sectors", "tess_total_points", "tess_flux_range"],
            "match_col": "tess_n_sectors",
            "summarize": lambda lc: vetting._summarize_flux_lc(lc, "sector", "tess_n_sectors", "tess_total_points", "tess_flux_range"),
        },
        "neowise": {
            "module": "NEOWISE LCs",
            "prefix": "neowise_lc",
            "summary_cols": ["neowise_n_epochs", "neowise_w1_range", "neowise_w2_range"],
            "match_col": "neowise_n_epochs",
            "summarize": vetting._summarize_neowise_lc,
        },
        "kepler": {
            "module": "Kepler LCs",
            "prefix": "kepler_lc",
            "summary_cols": ["kepler_n_quarters", "kepler_total_points", "kepler_flux_range"],
            "match_col": "kepler_n_quarters",
            "summarize": lambda lc: vetting._summarize_flux_lc(lc, "quarter", "kepler_n_quarters", "kepler_total_points", "kepler_flux_range"),
        },
        "aavso": {
            "module": "AAVSO LCs",
            "prefix": "aavso_lc",
            "summary_cols": ["aavso_lc_n_points"],
            "match_col": "aavso_lc_n_points",
            "summarize": lambda lc: vetting._summarize_count_lc(lc, "aavso_lc_n_points"),
        },
        "ogle": {
            "module": "OGLE LCs",
            "prefix": "ogle_lc",
            "summary_cols": ["ogle_lc_n_points", "ogle_lc_i_range", "ogle_lc_v_range"],
            "match_col": "ogle_lc_n_points",
            "summarize": vetting._summarize_ogle_lc,
        },
        "stripe82": {
            "module": "Stripe 82 LCs",
            "prefix": "stripe82_lc",
            "summary_cols": ["stripe82_lc_n_points", "stripe82_lc_u_range", "stripe82_lc_g_range", "stripe82_lc_r_range", "stripe82_lc_i_range", "stripe82_lc_z_range"],
            "match_col": "stripe82_lc_n_points",
            "summarize": vetting._summarize_stripe82_lc,
        },
        "allwise_mep": {
            "module": "AllWISE MEP LCs",
            "prefix": "allwise_mep_lc",
            "summary_cols": ["allwise_mep_n_epochs", "allwise_mep_w1_range", "allwise_mep_w2_range", "allwise_mep_w3_range", "allwise_mep_w4_range"],
            "match_col": "allwise_mep_n_epochs",
            "summarize": vetting._summarize_allwise_mep_lc,
        },
        "vvvx_virac": {
            "module": "VVVX/VIRAC2 LCs",
            "prefix": "vvvx_virac_lc",
            "summary_cols": ["vvvx_virac_n_epochs", "vvvx_virac_z_range", "vvvx_virac_y_range", "vvvx_virac_j_range", "vvvx_virac_h_range", "vvvx_virac_ks_range"],
            "match_col": "vvvx_virac_n_epochs",
            "summarize": vetting._summarize_vvvx_virac_lc,
        },
        "ps1": {
            "module": "Pan-STARRS LCs",
            "prefix": "ps1_lc",
            "summary_cols": ["ps1_lc_n_points"],
            "match_col": "ps1_lc_n_points",
            "summarize": lambda lc: vetting._summarize_count_lc(lc, "ps1_lc_n_points"),
        },
        "crts": {
            "module": "CRTS LCs",
            "prefix": "crts_lc",
            "summary_cols": ["crts_lc_n_points"],
            "match_col": "crts_lc_n_points",
            "summarize": lambda lc: vetting._summarize_count_lc(lc, "crts_lc_n_points"),
        },
    }


def _cache_only_default_value(col: str) -> object:
    return 0 if col.endswith(("_n_points", "_n_epochs", "_n_det", "_n_g", "_n_cyan", "_n_orange", "_n_sectors", "_n_quarters", "_total_points")) or col.endswith("_has_phot") else pd.NA


def _cache_only_status_summary(status_df: pd.DataFrame, module: str, candidate_id: str, summary_cols: list[str]) -> dict | None:
    if status_df.empty or not {"module", "candidate_id", "status"}.issubset(status_df.columns):
        return None
    rows = status_df[
        (status_df["module"].astype(str) == str(module))
        & (status_df["candidate_id"].astype(str) == str(candidate_id))
    ]
    if rows.empty:
        return None
    row = rows.iloc[-1]
    if str(row.get("status", "")) not in {"fetched", "no_data", "error", "failed"}:
        return None
    return {col: row.get(col, _cache_only_default_value(col)) for col in summary_cols}


def rebuild_external_lc_table_from_cache(df: pd.DataFrame, output_dir: Path, run_flags: dict[str, bool]) -> pd.DataFrame:
    from malca import vetting

    out = df.copy()
    specs = _cache_only_specs()
    status_df = vetting._read_external_lc_status(output_dir)
    messages: list[str] = []
    for key, enabled in run_flags.items():
        if not enabled or key not in specs:
            continue
        spec = specs[key]
        for col in spec["summary_cols"]:
            out[col] = _cache_only_default_value(col)
        found = 0
        positive = 0
        status_hits = 0
        for idx in out.index:
            cand_id = vetting._candidate_cache_id(out, idx)
            path = vetting._external_lc_path(output_dir, spec["prefix"], out, idx)
            summary = None
            lc_df = vetting._read_external_lc_file(path)
            if lc_df is not None:
                try:
                    summary = spec["summarize"](lc_df)
                except Exception:
                    summary = None
            if summary is None:
                summary = _cache_only_status_summary(status_df, spec["module"], cand_id, spec["summary_cols"])
                if summary is not None:
                    status_hits += 1
            if summary is None:
                continue
            found += 1
            for col in spec["summary_cols"]:
                out.loc[idx, col] = summary.get(col, _cache_only_default_value(col))
            try:
                if pd.notna(summary.get(spec["match_col"])) and float(summary.get(spec["match_col"])) > 0:
                    positive += 1
            except Exception:
                pass
        messages.append(f"{spec['module']}: restored {found} candidates from cache/status ({positive} with data; {status_hits} status-only)")
    for msg in messages:
        print(msg)
    return out


def _merge_into_review_db_with_retries(review_db: Path, merge_df: pd.DataFrame) -> int:
    lock_path = review_db.with_suffix(review_db.suffix + ".external_lcs_merge.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a", encoding="ascii") as lock_file:
        if fcntl is not None:
            print(f"Waiting for review DB merge lock: {lock_path}")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            attempts = 30
            for attempt in range(1, attempts + 1):
                try:
                    with closing(db_connect(review_db)) as conn:
                        return merge_candidate_results(conn, merge_df)
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower() or attempt >= attempts:
                        raise
                    delay = min(30.0, 2.0 * attempt)
                    print(
                        f"Review DB is locked; retrying merge in {delay:.0f}s "
                        f"({attempt}/{attempts - 1})"
                    )
                    time.sleep(delay)
            raise RuntimeError("Review DB merge retry loop exhausted")
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def run(args: argparse.Namespace) -> Path:
    input_path = args.input.expanduser()
    output_path = (args.output or _default_output_path(input_path)).expanduser()
    output_dir = (args.output_dir or input_path.parent).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    review_db_path = args.review_db.expanduser() if args.review_db else None

    checkpoint_path = None
    if not args.no_checkpoint:
        checkpoint_path = (args.checkpoint or _default_checkpoint_path(input_path, output_dir)).expanduser()

    df = with_feature_columns(
        _read_input_candidates(input_path),
        EXTERNAL_LC_INPUT_COLUMNS,
    )
    df = _ensure_candidate_id(df)
    if not getattr(args, "all_candidates", False):
        df = select_passing_candidates_if_present(df, printer=print)
    input_kind = "review DB" if _looks_like_review_db(input_path) else "candidate table"
    print(f"Loaded {len(df)} candidates from {input_kind}: {input_path}")
    print(f"Writing per-candidate LC files to {output_dir}")

    run_flags = _source_run_flags(args)
    if args.cache_only:
        print("Cache-only mode: rebuilding external-LC fields from existing parquet/status files; no remote lookups")
        out = rebuild_external_lc_table_from_cache(df, output_dir, run_flags)
    else:
        from malca.vetting import fetch_external_lcs

        out = fetch_external_lcs(
            df,
            output_dir=output_dir,
            run_atlas=args.run_atlas,
            run_ztf=not args.no_ztf,
            run_gaia_epoch=not args.no_gaia_epoch,
            run_tess=not args.no_tess,
            run_neowise=not args.no_neowise,
            run_kepler=not args.no_kepler,
            run_aavso=not args.no_aavso,
            run_ogle=not args.no_ogle,
            run_stripe82=not args.no_stripe82,
            run_allwise_mep=not args.no_allwise_mep,
            run_vvvx_virac=not args.no_vvvx_virac,
            run_ps1=not args.no_ps1,
            run_crts=not args.no_crts,
            atlas_token=args.atlas_token or os.environ.get("MALCA_ATLAS_TOKEN") or os.environ.get("ATLAS_API_TOKEN"),
            workers=args.workers,
            checkpoint_path=checkpoint_path,
            refresh_cache=args.refresh_cache,
        )

    write_feature_table(out, output_path)
    print(f"\nSaved external-LC table to {output_path}")

    if review_db_path:
        review_db = review_db_path
        merge_df = _merge_frame(out)
        updated = _merge_into_review_db_with_retries(review_db, merge_df)
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
