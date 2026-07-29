"""CLI for fetching external light-curve products for candidate tables."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import pandas as pd

from malca.enrichment.atlas_forced_photometry import ATLAS_SUMMARY_COLUMNS
from malca.external_lc_manifest import (
    index_external_lc_paths_from_manifest,
    upsert_external_lc_manifest_entry,
)
from malca.products.candidates import select_passing_candidates_if_present
from malca.products.feature_layers import feature_mapping_get, with_feature_columns
from malca.review.store import (
    db_connect,
    ensure_review_db_schema,
    merge_candidate_results,
)
from malca.io.table_io import read_feature_table, write_feature_table

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None


EXTERNAL_LC_PATTERNS = (
    "atlas_lc_*.parquet",
    "ztf_lc_*.parquet",
    "ztf_forced_lc_*.parquet",
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
    "superwasp_lc_*.parquet",
    "kelt_lc_*.parquet",
    "nsvs_lc_*.parquet",
    "asas3_lc_*.parquet",
    "crts_lc_*.parquet",
    "dasch_lc_*.parquet",
)

EXTERNAL_LC_COLUMNS = (
    "atlas_has_phot",
    "atlas_n_det_cyan",
    "atlas_n_det_orange",
    "atlas_cyan_range",
    "atlas_orange_range",
    "atlas_preprocess_version",
    "atlas_n_raw",
    "atlas_n_good",
    "atlas_n_rejected",
    "ztf_lc_n_det",
    "ztf_lc_g_range",
    "ztf_lc_r_range",
    "ztf_forced_lc_n_epochs",
    "ztf_forced_lc_n_good",
    "ztf_forced_lc_n_zg",
    "ztf_forced_lc_n_zr",
    "ztf_forced_lc_n_zi",
    "gaia_epoch_lc_n_g",
    "gaia_epoch_lc_g_range",
    "tess_n_sectors",
    "tess_total_points",
    "tess_flux_range",
    "tess_identity_status",
    "tess_identity_sep_arcsec",
    "tess_target_id",
    "neowise_n_epochs",
    "neowise_w1_range",
    "neowise_w2_range",
    "neowise_identity_status",
    "neowise_identity_sep_arcsec",
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
    "superwasp_lc_n_points",
    "superwasp_lc_time_span_days",
    "superwasp_lc_state",
    "kelt_lc_n_points",
    "kelt_lc_time_span_days",
    "kelt_lc_state",
    "nsvs_lc_n_points",
    "nsvs_lc_time_span_days",
    "nsvs_lc_state",
    "asas3_lc_n_points",
    "asas3_lc_time_span_days",
    "asas3_lc_state",
    "crts_lc_n_points",
    "crts_lc_time_span_days",
    "crts_lc_state",
    "dasch_lc_n_points",
    "dasch_lc_time_span_days",
    "dasch_lc_state",
)

EXTERNAL_LC_INPUT_COLUMNS = (
    "ra",
    "dec",
    "ra_deg",
    "dec_deg",
    "gaia_id",
    "source_id",
    "pmra",
    "pmdec",
    "ref_epoch",
    "tic_id",
    "ticid",
    "tess_id",
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

ZTF_FORCED_SUMMARY_COLUMNS = (
    "ztf_forced_lc_n_epochs",
    "ztf_forced_lc_n_good",
    "ztf_forced_lc_n_zg",
    "ztf_forced_lc_n_zr",
    "ztf_forced_lc_n_zi",
)

REVIEW_CLASS_BUCKETS = {
    "dipper": "Dipper",
    "ltv": "LTV",
    "microlensing": "Microlensing",
}


def _atlas_batch_size(value: str) -> int:
    size = int(value)
    if not 1 <= size <= 100:
        raise argparse.ArgumentTypeError("ATLAS batch size must be between 1 and 100")
    return size


def _ztf_forced_batch_size(value: str) -> int:
    size = int(value)
    if not 1 <= size <= 1500:
        raise argparse.ArgumentTypeError("ZTF forced-photometry batch size must be between 1 and 1500")
    return size


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
        "--ztf-forced",
        dest="run_ztf_forced",
        action="store_true",
        default=False,
        help="Enable resumable IPAC ZTF forced photometry (default: disabled)",
    )
    parser.add_argument(
        "--ztf-forced-only",
        action="store_true",
        help="Run only IPAC ZTF forced photometry; avoids ordinary catalog ZTF and other sources",
    )
    parser.add_argument("--ztf-forced-email", type=str, default=None, help="Registered ZFPS email (or MALCA_ZTF_FORCED_EMAIL)")
    parser.add_argument("--ztf-forced-userpass", type=str, default=None, help="Personal ZFPS password (or MALCA_ZTF_FORCED_USERPASS)")
    parser.add_argument("--ztf-forced-task-checkpoint", type=Path, default=None, help="ZTF forced-photometry request journal")
    parser.add_argument("--ztf-forced-batch-size", type=_ztf_forced_batch_size, default=1500, help="Coordinates per ZFPS request, at most 1500")
    parser.add_argument("--ztf-forced-jd-start", type=float, default=2458194.5, help="Earliest ZFPS epoch JD (default: survey start)")
    parser.add_argument("--ztf-forced-jd-end", type=float, default=None, help="Latest ZFPS epoch JD (default: 2026-01-01)")
    parser.add_argument("--ztf-forced-submit-only", action="store_true", help="Submit missing ZFPS batches without checking/downloading results")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for per-candidate LC parquet files (default: input parent; "
            "ATLAS-only review DB: <run>/results/external_lcs)"
        ),
    )
    parser.add_argument(
        "--review-db",
        type=Path,
        default=None,
        help=(
            "Optional review SQLite DB to merge enriched candidate fields into "
            "(ATLAS-only review DB input merges back automatically)"
        ),
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
    parser.add_argument(
        "--atlas-only",
        action="store_true",
        help=(
            "Run only the resumable ATLAS bulk fetcher; review-DB input defaults "
            "to the reviewed/keep/nonduplicate Dipper, LTV, and Microlensing cohort"
        ),
    )
    parser.add_argument(
        "--review-classes",
        nargs="+",
        choices=tuple(REVIEW_CLASS_BUCKETS),
        default=None,
        help=(
            "For review-DB input, select the reviewed/keep/nonduplicate publication "
            "cohort in these classes"
        ),
    )
    parser.add_argument(
        "--legacy-surveys-only",
        action="store_true",
        help=(
            "Run only SuperWASP, KELT, NSVS, ASAS-3, CRTS, and DASCH; "
            "review-DB input defaults to the reviewed Dipper cohort"
        ),
    )
    parser.add_argument(
        "--atlas-task-checkpoint",
        type=Path,
        default=None,
        help="Permanent per-candidate ATLAS task journal (default: <results>/external_lcs/atlas_forced_phot_tasks.parquet)",
    )
    parser.add_argument(
        "--atlas-batch-size",
        type=_atlas_batch_size,
        default=100,
        help="Coordinates per ATLAS list submission, at most 100 (default: 100)",
    )
    parser.add_argument(
        "--atlas-poll-interval",
        type=float,
        default=60.0,
        help="Seconds between ATLAS queue polls (default: 60)",
    )
    parser.add_argument(
        "--atlas-mjd-min",
        type=float,
        default=57000.0,
        help="Earliest ATLAS observation MJD (default: 57000)",
    )
    parser.add_argument("--atlas-mjd-max", type=float, default=None, help="Optional latest ATLAS observation MJD")
    parser.add_argument(
        "--atlas-image-type",
        choices=("reduced", "difference"),
        default="reduced",
        help="ATLAS target/reduced or difference-image photometry (default: reduced)",
    )
    parser.add_argument(
        "--atlas-max-wait",
        type=float,
        default=None,
        help="Optional total wait limit in seconds; unfinished jobs remain resumable, never no-data",
    )
    parser.add_argument(
        "--atlas-submit-only",
        action="store_true",
        help="Submit missing batches and exit; a later identical command polls saved task URLs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the cohort and print the ATLAS batch/output plan without network requests or writes",
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
    parser.add_argument("--no-superwasp", action="store_true", help="Skip SuperWASP light curves")
    parser.add_argument("--no-kelt", action="store_true", help="Skip KELT light curves")
    parser.add_argument("--no-nsvs", action="store_true", help="Skip NSVS light curves")
    parser.add_argument("--no-asas3", action="store_true", help="Skip ASAS-3 light curves")
    parser.add_argument("--no-crts", action="store_true", help="Skip CRTS light curves")
    parser.add_argument("--no-dasch", action="store_true", help="Skip DASCH light curves")
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


def _read_input_candidates(
    input_path: Path,
    *,
    review_classes: list[str] | None = None,
) -> pd.DataFrame:
    if not _looks_like_review_db(input_path):
        if review_classes:
            raise ValueError("--review-classes requires a review SQLite DB input")
        return read_feature_table(input_path)
    if review_classes:
        from malca.review.paper_candidates import load_reviewed_cohort

        buckets = [REVIEW_CLASS_BUCKETS[value] for value in review_classes]
        cohort = load_reviewed_cohort(
            input_path,
            buckets=buckets,
            only_reviewed=True,
            publication_only=True,
        )
        cohort = cohort.sort_values("candidate_id", kind="stable").reset_index(drop=True)
        return _hydrate_review_db_input(cohort, EXTERNAL_LC_INPUT_COLUMNS)
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


_CACHE_ONLY_TOUCHED_ATTR = "_external_lc_cache_only_touched"


def _source_run_flags(args: argparse.Namespace) -> dict[str, bool]:
    if getattr(args, "legacy_surveys_only", False):
        return {
            "atlas": False,
            "ztf_forced": False,
            "ztf": False,
            "gaia_epoch": False,
            "tess": False,
            "neowise": False,
            "kepler": False,
            "aavso": False,
            "ogle": False,
            "stripe82": False,
            "allwise_mep": False,
            "vvvx_virac": False,
            "ps1": False,
            "superwasp": not args.no_superwasp,
            "kelt": not args.no_kelt,
            "nsvs": not args.no_nsvs,
            "asas3": not args.no_asas3,
            "crts": not args.no_crts,
            "dasch": not args.no_dasch,
        }
    if getattr(args, "atlas_only", False) or getattr(args, "ztf_forced_only", False):
        return {
            "atlas": bool(getattr(args, "atlas_only", False)),
            "ztf_forced": bool(getattr(args, "ztf_forced_only", False)),
            "ztf": False,
            "gaia_epoch": False,
            "tess": False,
            "neowise": False,
            "kepler": False,
            "aavso": False,
            "ogle": False,
            "stripe82": False,
            "allwise_mep": False,
            "vvvx_virac": False,
            "ps1": False,
            "superwasp": False,
            "kelt": False,
            "nsvs": False,
            "asas3": False,
            "crts": False,
            "dasch": False,
        }
    return {
        "atlas": bool(args.run_atlas),
        "ztf_forced": bool(getattr(args, "run_ztf_forced", False)),
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
        "superwasp": not args.no_superwasp,
        "kelt": not args.no_kelt,
        "nsvs": not args.no_nsvs,
        "asas3": not args.no_asas3,
        "crts": not args.no_crts,
        "dasch": not args.no_dasch,
    }


def _cache_only_specs():
    from malca.enrichment import vetting

    return {
        "atlas": {
            "module": "ATLAS LCs",
            "prefix": "atlas_lc",
            "summary_cols": list(ATLAS_SUMMARY_COLUMNS),
            "match_col": "atlas_has_phot",
            "summarize": vetting._summarize_atlas_lc,
        },
        "ztf": {
            "module": "ZTF LCs",
            "prefix": "ztf_lc",
            "summary_cols": ["ztf_lc_n_det", "ztf_lc_g_range", "ztf_lc_r_range"],
            "match_col": "ztf_lc_n_det",
            "summarize": vetting._summarize_ztf_lc,
            "cache_key": lambda df, idx: vetting._coord_lookup_cache_key(
                df,
                idx,
                2.0,
                vetting.ZTF_LC_COLLECTION,
            ),
        },
        "gaia_epoch": {
            "module": "Gaia epoch LCs",
            "prefix": "gaia_epoch_lc",
            "summary_cols": ["gaia_epoch_lc_n_g", "gaia_epoch_lc_g_range"],
            "match_col": "gaia_epoch_lc_n_g",
            "summarize": vetting._summarize_gaia_epoch_lc,
            "cache_key": lambda df, idx: vetting._source_lookup_cache_key(
                vetting._parse_gaia_source_id_str(df.loc[idx, "gaia_id"] if "gaia_id" in df.columns else None),
                "gaia_epoch_lc",
            ),
        },
        "tess": {
            "module": "TESS LCs",
            "prefix": "tess_lc",
            "summary_cols": [
                "tess_n_sectors", "tess_total_points", "tess_flux_range",
                "tess_identity_status", "tess_identity_sep_arcsec", "tess_target_id",
            ],
            "match_col": "tess_n_sectors",
            "summarize": vetting._summarize_tess_lc,
            "cache_key": lambda df, idx: vetting._coord_lookup_cache_key(
                df,
                idx,
                vetting.TESS_SEARCH_RADIUS_ARCSEC,
                "tess",
            ),
        },
        "neowise": {
            "module": "NEOWISE LCs",
            "prefix": "neowise_lc",
            "summary_cols": [
                "neowise_n_epochs", "neowise_w1_range", "neowise_w2_range",
                "neowise_identity_status", "neowise_identity_sep_arcsec",
            ],
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
            "cache_key": lambda df, idx: vetting._coord_lookup_cache_key(
                df,
                idx,
                vetting.ALLWISE_MEP_MAX_SEP_ARCSEC,
                "allwise_mep",
            ),
        },
        "vvvx_virac": {
            "module": "VVVX/VIRAC2 LCs",
            "prefix": "vvvx_virac_lc",
            "summary_cols": ["vvvx_virac_n_epochs", "vvvx_virac_z_range", "vvvx_virac_y_range", "vvvx_virac_j_range", "vvvx_virac_h_range", "vvvx_virac_ks_range"],
            "match_col": "vvvx_virac_n_epochs",
            "summarize": vetting._summarize_vvvx_virac_lc,
            "cache_key": lambda df, idx: vetting._coord_lookup_cache_key(
                df,
                idx,
                vetting.VVVX_VIRAC_MAX_SEP_ARCSEC,
                "vvvx_virac2",
            ),
        },
        "ps1": {
            "module": "Pan-STARRS LCs",
            "prefix": "ps1_lc",
            "summary_cols": ["ps1_lc_n_points"],
            "match_col": "ps1_lc_n_points",
            "summarize": lambda lc: vetting._summarize_count_lc(lc, "ps1_lc_n_points"),
            "cache_key": lambda df, idx: vetting._coord_lookup_cache_key(
                df,
                idx,
                vetting.PANSTARRS_LC_RADIUS_DEG * 3600.0,
                "ps1_dr2",
            ),
        },
        "superwasp": {
            "module": "SuperWASP LCs",
            "prefix": "superwasp_lc",
            "summary_cols": [
                "superwasp_lc_n_points",
                "superwasp_lc_time_span_days",
                "superwasp_lc_state",
            ],
            "match_col": "superwasp_lc_n_points",
            "summarize": lambda lc: vetting._summarize_legacy_mag_lc(
                lc,
                "superwasp_lc",
                preferred_proc_type="raw",
            ),
            "cache_key": lambda df, idx: vetting._coord_lookup_cache_key(
                df,
                idx,
                vetting.legacy_survey_lcs.SUPERWASP_MATCH_RADIUS_ARCSEC,
                "superwasp_lc",
            ),
        },
        "kelt": {
            "module": "KELT LCs",
            "prefix": "kelt_lc",
            "summary_cols": [
                "kelt_lc_n_points",
                "kelt_lc_time_span_days",
                "kelt_lc_state",
            ],
            "match_col": "kelt_lc_n_points",
            "summarize": lambda lc: vetting._summarize_legacy_mag_lc(
                lc,
                "kelt_lc",
                preferred_proc_type="raw",
            ),
            "cache_key": lambda df, idx: vetting._coord_lookup_cache_key(
                df,
                idx,
                vetting.legacy_survey_lcs.KELT_MATCH_RADIUS_ARCSEC,
                "kelt_lc",
            ),
        },
        "nsvs": {
            "module": "NSVS LCs",
            "prefix": "nsvs_lc",
            "summary_cols": [
                "nsvs_lc_n_points",
                "nsvs_lc_time_span_days",
                "nsvs_lc_state",
            ],
            "match_col": "nsvs_lc_n_points",
            "summarize": lambda lc: vetting._summarize_legacy_mag_lc(
                lc,
                "nsvs_lc",
            ),
            "cache_key": lambda df, idx: vetting._coord_lookup_cache_key(
                df,
                idx,
                vetting.legacy_survey_lcs.NSVS_MATCH_RADIUS_ARCSEC,
                "nsvs_lc",
            ),
        },
        "asas3": {
            "module": "ASAS-3 LCs",
            "prefix": "asas3_lc",
            "summary_cols": [
                "asas3_lc_n_points",
                "asas3_lc_time_span_days",
                "asas3_lc_state",
            ],
            "match_col": "asas3_lc_n_points",
            "summarize": lambda lc: vetting._summarize_legacy_mag_lc(
                lc,
                "asas3_lc",
                selected_only=True,
            ),
            "cache_key": lambda df, idx: vetting._coord_lookup_cache_key(
                df,
                idx,
                vetting.legacy_survey_lcs.ASAS3_MATCH_RADIUS_ARCSEC,
                "asas3_lc",
            ),
        },
        "crts": {
            "module": "CRTS LCs",
            "prefix": "crts_lc",
            "summary_cols": [
                "crts_lc_n_points",
                "crts_lc_time_span_days",
                "crts_lc_state",
            ],
            "match_col": "crts_lc_n_points",
            "summarize": lambda lc: vetting._summarize_legacy_mag_lc(
                lc,
                "crts_lc",
                time_col="mjd",
            ),
            "cache_key": lambda df, idx: vetting._coord_lookup_cache_key(
                df,
                idx,
                vetting.CRTS_MATCH_RADIUS_ARCSEC,
                "crts",
            ),
        },
        "dasch": {
            "module": "DASCH LCs",
            "prefix": "dasch_lc",
            "summary_cols": [
                "dasch_lc_n_points",
                "dasch_lc_time_span_days",
                "dasch_lc_state",
            ],
            "match_col": "dasch_lc_n_points",
            "summarize": lambda lc: vetting._summarize_legacy_mag_lc(
                lc,
                "dasch_lc",
            ),
            "cache_key": lambda df, idx: vetting._coord_lookup_cache_key(
                df,
                idx,
                vetting.legacy_survey_lcs.DASCH_MATCH_RADIUS_ARCSEC,
                "dasch_lc",
            ),
        },
    }


def _cache_only_default_value(col: str) -> object:
    if col.endswith("_has_phot"):
        return False
    if col in {"atlas_n_raw", "atlas_n_good", "atlas_n_rejected"}:
        return 0
    return 0 if col.endswith(("_n_points", "_n_epochs", "_n_det", "_n_g", "_n_cyan", "_n_orange", "_n_sectors", "_n_quarters", "_total_points")) else pd.NA


def _cache_only_status_summary(status_df: pd.DataFrame, module: str, candidate_id: str, summary_cols: list[str]) -> dict | None:
    if status_df.empty or not {"module", "candidate_id", "status"}.issubset(status_df.columns):
        return None
    rows = status_df[
        (status_df["module"].astype(str) == str(module))
        & (status_df["candidate_id"].astype(str) == str(candidate_id))
    ]
    if rows.empty:
        return None
    if "updated_unix" in rows.columns:
        rows = rows.sort_values("updated_unix", kind="stable")
    row = rows.iloc[-1]
    if str(row.get("status", "")) not in {"fetched", "no_data", "error", "failed", "identity_unverified"}:
        return None
    summary = {
        col: row.get(col, _cache_only_default_value(col))
        for col in summary_cols
    }
    status = str(row.get("status", ""))
    for col in summary_cols:
        if not col.endswith("_state"):
            continue
        value = summary.get(col)
        if value is not None and not pd.isna(value) and str(value).strip():
            continue
        if status in {"error", "failed"}:
            summary[col] = "fetch_failed"
        elif status == "no_data":
            summary[col] = "no_coverage"
        elif status == "fetched":
            summary[col] = "matched"
    return summary


def _read_external_lc_statuses(vetting, output_dir: Path) -> pd.DataFrame:
    root = Path(output_dir)
    status_name = str(getattr(vetting, "EXTERNAL_LC_STATUS_FILE", "_external_lc_status.parquet"))
    paths: list[Path] = []
    direct = root / status_name
    if direct.exists():
        paths.append(direct)
    if root.exists():
        for path in sorted(root.rglob(status_name)):
            if path not in paths:
                paths.append(path)

    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            status = pd.read_parquet(path)
        except Exception as exc:
            print(f"External LC status warning: could not read {path}: {exc}")
            continue
        if status.empty:
            continue
        status = status.copy()
        status["_status_path"] = str(path)
        frames.append(status)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _cache_only_lc_path(vetting, output_dir: Path, manifest_paths: dict[str, str], spec: dict, df: pd.DataFrame, idx: object) -> Path | None:
    cand_id = vetting._candidate_cache_id(df, idx)
    path_text = manifest_paths.get(str(cand_id))
    if path_text:
        return Path(path_text)
    return vetting._external_lc_path(output_dir, spec["prefix"], df, idx)


def _read_cache_only_lc(vetting, source_key: str, path: Path | None) -> pd.DataFrame | None:
    if source_key != "atlas":
        return vetting._read_external_lc_file(path)
    if path is None or not path.exists():
        return None
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return None
    if not frame.empty:
        return frame
    # The new fetcher deliberately writes a valid zero-row parquet for a
    # completed header-only ATLAS result.  Require its persisted mode
    # provenance so an arbitrary legacy empty file cannot clear review data.
    modes = frame.attrs.get("atlas_image_types")
    single_mode = frame.attrs.get("atlas_image_type")
    has_mode_provenance = (
        isinstance(modes, (list, tuple, set)) and bool(modes)
    ) or (isinstance(single_mode, str) and bool(single_mode.strip()))
    return frame if has_mode_provenance else None


def _cache_only_source_merge_frames(out: pd.DataFrame, run_flags: dict[str, bool]) -> list[tuple[str, pd.DataFrame]]:
    specs = _cache_only_specs()
    touched = out.attrs.get(_CACHE_ONLY_TOUCHED_ATTR, {})
    if not isinstance(touched, dict):
        touched = {}

    id_cols = [c for c in ("candidate_id", "asas_sn_id") if c in out.columns]
    if not id_cols:
        return []

    frames: list[tuple[str, pd.DataFrame]] = []
    candidate_ids = out["candidate_id"].astype(str) if "candidate_id" in out.columns else pd.Series("", index=out.index)
    for key, enabled in run_flags.items():
        if not enabled or key not in specs:
            continue
        touched_ids = {str(v) for v in touched.get(key, []) if str(v)}
        if not touched_ids:
            continue
        value_cols = [c for c in specs[key]["summary_cols"] if c in out.columns]
        if not value_cols:
            continue
        frame = out.loc[candidate_ids.isin(touched_ids), id_cols + value_cols].copy()
        if frame.empty:
            continue
        frame = frame.drop_duplicates(subset=id_cols[0], keep="last")
        frames.append((key, frame))
    return frames


def _cache_only_summary_is_positive(summary: dict, match_col: str) -> bool:
    value = summary.get(match_col)
    if isinstance(value, bool):
        return value
    try:
        return bool(pd.notna(value) and float(value) > 0)
    except Exception:
        return False


def _cache_only_status_row(vetting, spec: dict, df: pd.DataFrame, idx: object, summary: dict) -> dict | None:
    cache_key_func = spec.get("cache_key")
    if cache_key_func is None:
        return None
    try:
        cache_key = cache_key_func(df, idx)
    except Exception:
        cache_key = None
    if cache_key is None:
        return None
    identity_values = [
        str(value)
        for key, value in summary.items()
        if key.endswith("_identity_status") and pd.notna(value)
    ]
    state_values = {
        str(value)
        for key, value in summary.items()
        if key.endswith("_state") and pd.notna(value)
    }
    positive = _cache_only_summary_is_positive(summary, spec["match_col"])
    identity_unverified = positive and any(value != "matched" for value in identity_values)
    if "fetch_failed" in state_values:
        status = "error"
    elif identity_unverified:
        status = "identity_unverified"
    else:
        status = "fetched" if positive else "no_data"
    if hasattr(vetting, "_external_lc_status_row"):
        return vetting._external_lc_status_row(
            df,
            idx,
            module=spec["module"],
            cache_key=cache_key,
            summary=summary,
            status=status,
        )
    return {
        "module": spec["module"],
        "candidate_id": vetting._candidate_cache_id(df, idx),
        "cache_key": cache_key,
        "status": status,
        "updated_unix": time.time(),
        **summary,
    }


def rebuild_external_lc_table_from_cache(
    df: pd.DataFrame,
    output_dir: Path,
    run_flags: dict[str, bool],
    *,
    results_root: Path | None = None,
) -> pd.DataFrame:
    from malca.enrichment import vetting

    out = df.copy()
    specs = _cache_only_specs()
    status_df = _read_external_lc_statuses(vetting, output_dir)
    messages: list[str] = []
    repaired_status_rows: list[dict] = []
    repaired_manifest_rows = 0
    touched_by_source: dict[str, set[str]] = {}
    for key, enabled in run_flags.items():
        if not enabled or key not in specs:
            continue
        spec = specs[key]
        for col in spec["summary_cols"]:
            out[col] = _cache_only_default_value(col)
        manifest_paths = index_external_lc_paths_from_manifest(
            str(Path(results_root or output_dir).expanduser()),
            spec["prefix"],
        )
        touched_by_source[key] = set()
        found = 0
        positive = 0
        status_hits = 0
        for idx in out.index:
            cand_id = vetting._candidate_cache_id(out, idx)
            path = _cache_only_lc_path(vetting, output_dir, manifest_paths, spec, out, idx)
            summary = None
            summary_from_status = False
            lc_df = _read_cache_only_lc(vetting, key, path)
            if lc_df is not None:
                if (
                    path is not None
                    and str(cand_id) not in manifest_paths
                    and upsert_external_lc_manifest_entry(
                        results_root or output_dir,
                        candidate_id=str(cand_id),
                        source=key,
                        file_prefix=spec["prefix"],
                        path=path,
                    )
                ):
                    manifest_paths[str(cand_id)] = str(path)
                    repaired_manifest_rows += 1
                try:
                    summary = spec["summarize"](lc_df)
                except Exception:
                    summary = None
                if summary is not None:
                    row = _cache_only_status_row(vetting, spec, out, idx, summary)
                    if row is not None:
                        repaired_status_rows.append(row)
            if summary is None:
                summary = _cache_only_status_summary(status_df, spec["module"], cand_id, spec["summary_cols"])
                if summary is not None:
                    status_hits += 1
                    summary_from_status = True
            if summary is None:
                continue
            # Historical status rows predate identity provenance.  Counts can
            # be restored in cache-only mode, but they must be labelled as
            # unverified rather than silently appearing equivalent to a newly
            # matched TESS/NEOWISE light curve.
            for identity_col in (
                col for col in spec["summary_cols"] if col.endswith("_identity_status")
            ):
                value = summary.get(identity_col)
                if value is None or pd.isna(value) or not str(value).strip():
                    summary[identity_col] = "legacy_unverified"
            if summary_from_status:
                # Persist normalized fields inferred from historical terminal
                # rows (for example, ``status=no_data`` -> ``no_coverage``),
                # rather than exposing them only in the rebuilt table.
                row = _cache_only_status_row(vetting, spec, out, idx, summary)
                if row is not None:
                    repaired_status_rows.append(row)
            touched_by_source[key].add(str(cand_id))
            found += 1
            for col in spec["summary_cols"]:
                out.loc[idx, col] = summary.get(col, _cache_only_default_value(col))
            if _cache_only_summary_is_positive(summary, spec["match_col"]):
                positive += 1
        messages.append(f"{spec['module']}: restored {found} candidates from cache/status ({positive} with data; {status_hits} status-only)")
    if repaired_status_rows:
        vetting._write_external_lc_status(output_dir, repaired_status_rows)
        messages.append(f"External LC status: repaired {len(repaired_status_rows)} rows from existing LC files")
    if repaired_manifest_rows:
        messages.append(f"External LC manifest: indexed {repaired_manifest_rows} existing LC files")
    for msg in messages:
        print(msg)
    out.attrs[_CACHE_ONLY_TOUCHED_ATTR] = {key: sorted(values) for key, values in touched_by_source.items() if values}
    return out


def _merge_into_review_db_with_retries(
    review_db: Path,
    merge_df: pd.DataFrame,
    *,
    clear_columns: tuple[str, ...] = (),
) -> int:
    lock_path = review_db.with_suffix(review_db.suffix + ".external_lcs_merge.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a", encoding="ascii") as lock_file:
        if fcntl is not None:
            print(f"Waiting for review DB merge lock: {lock_path}")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            ensure_review_db_schema(review_db)
            attempts = 30
            for attempt in range(1, attempts + 1):
                try:
                    with closing(db_connect(review_db)) as conn:
                        return merge_candidate_results(
                            conn,
                            merge_df,
                            clear_columns=clear_columns,
                        )
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


def _merge_source_frames_into_review_db_with_retries(review_db: Path, source_frames: list[tuple[str, pd.DataFrame]]) -> dict[str, int]:
    source_frames = [(key, frame) for key, frame in source_frames if frame is not None and not frame.empty]
    if not source_frames:
        return {}

    lock_path = review_db.with_suffix(review_db.suffix + ".external_lcs_merge.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a", encoding="ascii") as lock_file:
        if fcntl is not None:
            print(f"Waiting for review DB merge lock: {lock_path}")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            ensure_review_db_schema(review_db)
            attempts = 30
            for attempt in range(1, attempts + 1):
                try:
                    updates: dict[str, int] = {}
                    with closing(db_connect(review_db)) as conn:
                        for source_key, frame in source_frames:
                            updates[source_key] = merge_candidate_results(
                                conn,
                                frame,
                                clear_columns=(
                                    ATLAS_SUMMARY_COLUMNS if source_key == "atlas" else ()
                                ),
                            )
                    return updates
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


def _atlas_results_root(input_path: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        directory = output_dir.expanduser()
        return directory.parent if directory.name == "external_lcs" else directory
    if _looks_like_review_db(input_path) and input_path.parent.name == "review":
        return input_path.parent.parent / "results"
    return input_path.parent


def _atlas_terminal_merge_frame(out: pd.DataFrame) -> pd.DataFrame:
    id_cols = [column for column in ("candidate_id", "asas_sn_id") if column in out.columns]
    value_cols = [column for column in ATLAS_SUMMARY_COLUMNS if column in out.columns]
    if not id_cols or "atlas_has_phot" not in value_cols:
        return pd.DataFrame(columns=[*id_cols, *value_cols])
    terminal = out["atlas_has_phot"].notna()
    frame = out.loc[terminal, [*id_cols, *value_cols]].copy()
    if frame.empty:
        return frame
    return frame.drop_duplicates(subset=id_cols[0], keep="last")


def run(args: argparse.Namespace) -> Path:
    input_path = args.input.expanduser()
    atlas_only = bool(getattr(args, "atlas_only", False))
    legacy_surveys_only = bool(getattr(args, "legacy_surveys_only", False))
    # ZFPS is an asynchronous bulk service, so even its non-"only" spelling
    # intentionally takes the dedicated forced-photometry path.
    ztf_forced_only = bool(getattr(args, "ztf_forced_only", False) or getattr(args, "run_ztf_forced", False))
    if sum((atlas_only, ztf_forced_only, legacy_surveys_only)) > 1:
        raise ValueError(
            "--atlas-only, --ztf-forced-only, and --legacy-surveys-only "
            "are mutually exclusive"
        )
    forced_only = atlas_only or ztf_forced_only
    review_classes = getattr(args, "review_classes", None)
    all_candidates = bool(getattr(args, "all_candidates", False))
    if (
        atlas_only
        and review_classes is None
        and not all_candidates
        and _looks_like_review_db(input_path)
    ):
        review_classes = list(REVIEW_CLASS_BUCKETS)
    if (
        ztf_forced_only
        and review_classes is None
        and not all_candidates
        and _looks_like_review_db(input_path)
    ):
        review_classes = ["dipper"]
    if (
        legacy_surveys_only
        and review_classes is None
        and not all_candidates
        and _looks_like_review_db(input_path)
    ):
        review_classes = ["dipper"]
    if forced_only or legacy_surveys_only:
        results_root = _atlas_results_root(input_path, args.output_dir)
        output_dir = (args.output_dir or results_root / "external_lcs").expanduser()
        if atlas_only:
            default_name = "atlas_reviewed_events_external_lcs.parquet"
        elif ztf_forced_only:
            default_name = "ztf_forced_reviewed_dippers_external_lcs.parquet"
        else:
            default_name = "legacy_surveys_reviewed_dippers_external_lcs.parquet"
        output_path = (
            args.output or results_root / default_name
        ).expanduser()
    else:
        atlas_review_db_defaults = (
            bool(getattr(args, "run_atlas", False))
            and args.output_dir is None
            and _looks_like_review_db(input_path)
        )
        if atlas_review_db_defaults:
            default_results_root = _atlas_results_root(input_path, None)
            output_dir = default_results_root / "external_lcs"
            output_path = (
                args.output or default_results_root / f"{input_path.stem}_external_lcs.parquet"
            ).expanduser()
        else:
            output_dir = (args.output_dir or input_path.parent).expanduser()
            output_path = (args.output or _default_output_path(input_path)).expanduser()
        results_root = _atlas_results_root(input_path, output_dir)

    review_db_path = args.review_db.expanduser() if args.review_db else None
    if (forced_only or legacy_surveys_only) and review_db_path is None and _looks_like_review_db(input_path):
        review_db_path = input_path

    checkpoint_path = None
    if not args.no_checkpoint and not forced_only:
        checkpoint_path = (args.checkpoint or _default_checkpoint_path(input_path, output_dir)).expanduser()

    df = with_feature_columns(
        _read_input_candidates(
            input_path,
            review_classes=review_classes,
        ),
        EXTERNAL_LC_INPUT_COLUMNS,
    )
    df = _ensure_candidate_id(df)
    if not all_candidates and not review_classes:
        df = select_passing_candidates_if_present(df, printer=print)
    input_kind = "review DB" if _looks_like_review_db(input_path) else "candidate table"
    print(f"Loaded {len(df)} candidates from {input_kind}: {input_path}")
    print(f"Writing per-candidate LC files to {output_dir}")

    run_flags = _source_run_flags(args)
    if getattr(args, "dry_run", False):
        valid_coordinates = (
            pd.to_numeric(df.get("ra"), errors="coerce").notna()
            & pd.to_numeric(df.get("dec"), errors="coerce").notna()
        )
        if "review_bucket" in df.columns:
            counts = df["review_bucket"].value_counts().reindex(
                [REVIEW_CLASS_BUCKETS[key] for key in (review_classes or [])],
                fill_value=0,
            )
            print("Review cohort: " + ", ".join(f"{name}={int(count)}" for name, count in counts.items()))
        n_valid = int(valid_coordinates.sum())
        if legacy_surveys_only:
            print(
                "Legacy survey dry run: "
                f"{n_valid} coordinate(s) across SuperWASP, KELT, NSVS, "
                "ASAS-3, CRTS, and DASCH"
            )
        else:
            batch_size = int(args.ztf_forced_batch_size if ztf_forced_only else args.atlas_batch_size)
            source = "ZTF forced photometry" if ztf_forced_only else "ATLAS"
            print(f"{source} dry run: {n_valid} coordinate(s) in {math.ceil(n_valid / batch_size) if n_valid else 0} batch(es) of at most {batch_size}")
            journal = args.ztf_forced_task_checkpoint or output_dir / "ztf_forced_phot_tasks.parquet" if ztf_forced_only else args.atlas_task_checkpoint or output_dir / "atlas_forced_phot_tasks.parquet"
            print(f"Task journal: {journal}")
        print(f"Manifest: {results_root / 'external_lc_manifest.parquet'}")
        print(f"Summary table: {output_path}")
        return output_path

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_only:
        print("Cache-only mode: rebuilding external-LC fields from existing parquet/status files; no remote lookups")
        out = rebuild_external_lc_table_from_cache(
            df,
            output_dir,
            run_flags,
            results_root=results_root,
        )
    elif atlas_only:
        from malca.enrichment.atlas_forced_photometry import query_atlas_forced_phot

        out = query_atlas_forced_phot(
            df,
            token=args.atlas_token or os.environ.get("MALCA_ATLAS_TOKEN") or os.environ.get("ATLAS_API_TOKEN"),
            output_dir=output_dir,
            results_root=results_root,
            refresh_cache=args.refresh_cache,
            task_checkpoint=args.atlas_task_checkpoint,
            batch_size=args.atlas_batch_size,
            poll_interval=args.atlas_poll_interval,
            mjd_min=args.atlas_mjd_min,
            mjd_max=args.atlas_mjd_max,
            image_type=args.atlas_image_type,
            max_wait_seconds=args.atlas_max_wait,
            submit_only=args.atlas_submit_only,
            progress=print,
        )
    elif ztf_forced_only:
        from malca.enrichment.ztf_forced_photometry import query_ztf_forced_phot

        out = query_ztf_forced_phot(
            df,
            email=args.ztf_forced_email,
            userpass=args.ztf_forced_userpass,
            output_dir=output_dir,
            results_root=results_root,
            task_checkpoint=args.ztf_forced_task_checkpoint,
            batch_size=args.ztf_forced_batch_size,
            jd_start=args.ztf_forced_jd_start,
            jd_end=args.ztf_forced_jd_end,
            submit_only=args.ztf_forced_submit_only,
            refresh_cache=args.refresh_cache,
            progress=print,
        )
    else:
        from malca.enrichment.vetting import fetch_external_lcs

        out = fetch_external_lcs(
            df,
            output_dir=output_dir,
            run_atlas=run_flags["atlas"],
            run_ztf=run_flags["ztf"],
            run_gaia_epoch=run_flags["gaia_epoch"],
            run_tess=run_flags["tess"],
            run_neowise=run_flags["neowise"],
            run_kepler=run_flags["kepler"],
            run_aavso=run_flags["aavso"],
            run_ogle=run_flags["ogle"],
            run_stripe82=run_flags["stripe82"],
            run_allwise_mep=run_flags["allwise_mep"],
            run_vvvx_virac=run_flags["vvvx_virac"],
            run_ps1=run_flags["ps1"],
            run_superwasp=run_flags["superwasp"],
            run_kelt=run_flags["kelt"],
            run_nsvs=run_flags["nsvs"],
            run_asas3=run_flags["asas3"],
            run_crts=run_flags["crts"],
            run_dasch=run_flags["dasch"],
            atlas_token=args.atlas_token or os.environ.get("MALCA_ATLAS_TOKEN") or os.environ.get("ATLAS_API_TOKEN"),
            atlas_results_root=results_root,
            atlas_task_checkpoint=args.atlas_task_checkpoint,
            atlas_batch_size=args.atlas_batch_size,
            atlas_poll_interval=args.atlas_poll_interval,
            atlas_mjd_min=args.atlas_mjd_min,
            atlas_mjd_max=args.atlas_mjd_max,
            atlas_image_type=args.atlas_image_type,
            atlas_max_wait_seconds=args.atlas_max_wait,
            atlas_submit_only=args.atlas_submit_only,
            workers=args.workers,
            checkpoint_path=checkpoint_path,
            refresh_cache=args.refresh_cache,
        )

    write_feature_table(out, output_path)
    print(f"\nSaved external-LC table to {output_path}")

    if review_db_path:
        review_db = review_db_path
        if atlas_only:
            if args.cache_only:
                source_frames = dict(_cache_only_source_merge_frames(out, run_flags))
                merge_df = source_frames.get("atlas", pd.DataFrame())
            else:
                merge_df = _atlas_terminal_merge_frame(out)
            updated = _merge_into_review_db_with_retries(
                review_db,
                merge_df,
                clear_columns=ATLAS_SUMMARY_COLUMNS,
            )
            print(
                f"Merged terminal ATLAS fields into {review_db} "
                f"({updated} candidates updated; queued jobs left untouched)"
            )
        elif ztf_forced_only:
            value_cols = [column for column in ZTF_FORCED_SUMMARY_COLUMNS if column in out.columns]
            terminal = out.loc[out.get("ztf_forced_lc_n_epochs", pd.Series(index=out.index)).notna(), [*(["candidate_id"] if "candidate_id" in out else []), *value_cols]]
            updated = _merge_into_review_db_with_retries(review_db, terminal, clear_columns=ZTF_FORCED_SUMMARY_COLUMNS)
            print(f"Merged downloaded ZTF forced-photometry fields into {review_db} ({updated} candidates updated; pending jobs left untouched)")
        elif args.cache_only:
            source_frames = _cache_only_source_merge_frames(out, run_flags)
            updates = _merge_source_frames_into_review_db_with_retries(review_db, source_frames)
            updated = sum(updates.values())
            source_count = sum(1 for value in updates.values() if value)
            print(f"Merged cache-only external-LC fields into {review_db} ({updated} source-candidate updates across {source_count} sources)")
        else:
            merge_df = _merge_frame(out)
            if run_flags.get("atlas"):
                other_df = merge_df.drop(
                    columns=[column for column in ATLAS_SUMMARY_COLUMNS if column in merge_df.columns]
                )
                other_updated = _merge_into_review_db_with_retries(review_db, other_df)
                atlas_df = _atlas_terminal_merge_frame(out)
                atlas_updated = _merge_into_review_db_with_retries(
                    review_db,
                    atlas_df,
                    clear_columns=ATLAS_SUMMARY_COLUMNS,
                )
                print(
                    f"Merged external-LC fields into {review_db} "
                    f"({other_updated} general candidate updates; "
                    f"{atlas_updated} terminal ATLAS updates)"
                )
            else:
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
