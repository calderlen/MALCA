#!/usr/bin/env python3
"""Audit 12-15 mag v2 candidates against Brayden reproduction results."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from malca.table_io import read_feature_table

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from malca.evaluation.validation import DEFAULT_CANDIDATES as BRAYDEN_CANDIDATES
from malca.config import DEFAULT_OUTPUT_DIR as MALCA_DEFAULT_OUTPUT_DIR


DEFAULT_CANDIDATES_CSV = Path("output/12-15mag_candidates_v2.csv")
DEFAULT_REPRODUCTION_DIR = MALCA_DEFAULT_OUTPUT_DIR / "logs" / "reproduction"
DEFAULT_OUTPUT_DIR = MALCA_DEFAULT_OUTPUT_DIR / "audits" / "12_15_v2_candidate_discrepancies"
DEFAULT_PROVENANCE_TABLES = (
    Path("output/runs/12_12.5_may13testrun/results/lc_events_enriched.parquet"),
    Path("output/runs/output_bundle_12.5_13_home_bundle_12.5_13/results/lc_events_enriched.parquet"),
    Path("output/runs/output_bundle_13_13.5_bundle_13_13.5/results/lc_events_enriched.parquet"),
    Path("output/runs/output_bundle_13.5_14_bundle_13.5_14/results/lc_events_enriched.parquet"),
    Path("output/runs/output_bundle_14_14.5_bundle_14_14.5/results/lc_events_enriched.parquet"),
    Path("output/runs/output_bundle_14.5_15_bundle_14.5_15/results/lc_events_enriched.parquet"),
    Path("output/runs/runs_march18_bundle_all/results/lc_events_enriched_all.parquet"),
)
REQUIRED_CANDIDATE_COLUMNS = ("path", "asas_sn_id", "mag_bin")
INTEGER_FLOAT_RE = re.compile(r"^([+-]?\d+)\.0+$")


def normalize_id(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>", "null"}:
        return ""
    match = INTEGER_FLOAT_RE.match(text)
    if match:
        text = match.group(1)
    return text.casefold()


def finite_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>", "null"}:
        return ""
    return text


def parse_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().casefold()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n"}:
        return False
    return None


def id_from_path(value: object) -> str:
    text = finite_text(value)
    if not text:
        return ""
    return Path(text).stem


def find_latest_reproduction_csv(reproduction_dir: Path = DEFAULT_REPRODUCTION_DIR) -> Path:
    files = sorted(reproduction_dir.glob("reproduction_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No reproduction_*.csv files found under {reproduction_dir}")
    return files[-1]


def load_primary_candidates(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Candidate CSV not found: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing_columns = [column for column in REQUIRED_CANDIDATE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Candidate CSV missing required columns: {', '.join(missing_columns)}")

    empty_columns = []
    for column in REQUIRED_CANDIDATE_COLUMNS:
        if df[column].map(finite_text).eq("").any():
            empty_columns.append(column)
    if empty_columns:
        raise ValueError(f"Candidate CSV has empty required values in: {', '.join(empty_columns)}")

    out = df.copy()
    out["asas_sn_id_norm"] = out["asas_sn_id"].map(normalize_id)
    if out["asas_sn_id_norm"].eq("").any():
        raise ValueError("Candidate CSV has empty normalized asas_sn_id values.")

    duplicates = out.loc[out["asas_sn_id_norm"].duplicated(keep=False), "asas_sn_id"].tolist()
    if duplicates:
        sample = ", ".join(map(str, duplicates[:10]))
        raise ValueError(f"Candidate CSV has duplicate normalized asas_sn_id values: {sample}")

    return out


def load_reproduction_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Reproduction CSV not found: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "source_id" not in df.columns:
        raise ValueError("Reproduction CSV missing required column: source_id")
    if "detected" not in df.columns:
        raise ValueError("Reproduction CSV missing required column: detected")

    out = df.copy()
    out["source_id_norm"] = out["source_id"].map(normalize_id)
    if out["source_id_norm"].eq("").any():
        raise ValueError("Reproduction CSV has empty normalized source_id values.")
    if out["source_id_norm"].duplicated().any():
        duplicates = out.loc[out["source_id_norm"].duplicated(keep=False), "source_id"].tolist()
        sample = ", ".join(map(str, duplicates[:10]))
        raise ValueError(f"Reproduction CSV has duplicate normalized source_id values: {sample}")
    out["reproduction_detected"] = out["detected"].map(parse_bool)
    if out["reproduction_detected"].isna().any():
        bad = out.loc[out["reproduction_detected"].isna(), "detected"].unique().tolist()
        raise ValueError(f"Could not parse reproduction detected values: {bad}")
    return out


def brayden_candidates_frame(candidates: Iterable[dict[str, object]] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(list(candidates or BRAYDEN_CANDIDATES)).copy()
    if "source_id" not in df.columns:
        raise ValueError("Brayden candidates must include source_id.")
    if "mag_bin" not in df.columns:
        raise ValueError("Brayden candidates must include mag_bin.")
    out = df.rename(columns={"mag_bin": "expected_mag_bin"})
    out["source_id"] = out["source_id"].astype(str)
    out["source_id_norm"] = out["source_id"].map(normalize_id)
    out["expected_detected"] = out["expected_detected"].map(parse_bool)
    if out["source_id_norm"].duplicated().any():
        raise ValueError("Brayden candidates contain duplicate normalized source_id values.")
    return out


def classify_discrepancy(row: pd.Series) -> str:
    present = bool(row["in_12_15_v2"])
    detected = parse_bool(row.get("reproduction_detected"))
    expected = parse_bool(row.get("expected_detected"))
    expected_mag_bin = finite_text(row.get("expected_mag_bin"))
    v2_mag_bin = finite_text(row.get("v2_mag_bin"))

    if present and expected_mag_bin and v2_mag_bin and expected_mag_bin != v2_mag_bin:
        return "mag_bin_mismatch"
    if detected is True and not present:
        return "reproduced_detected_missing_from_v2"
    if detected is False and present:
        return "reproduced_rejected_present_in_v2"
    if expected is True and not present:
        return "expected_detected_missing_from_v2"
    if present and detected is True:
        return "consistent_present_detected"
    if not present and detected is False:
        return "consistent_absent_rejected"
    if present:
        return "present_without_reproduction_detection"
    return "missing_without_reproduction_detection"


def candidate_counts_by_mag_bin(candidates: pd.DataFrame) -> pd.DataFrame:
    return (
        candidates.groupby("mag_bin", dropna=False)
        .size()
        .rename("n_candidates")
        .reset_index()
        .sort_values("mag_bin")
        .reset_index(drop=True)
    )


def read_provenance_ids(path: Path) -> pd.DataFrame:
    for column in ("asas_sn_id", "candidate_id", "source_id", "path"):
        try:
            df = read_feature_table(path, columns=[column])
        except Exception:
            continue
        out = pd.DataFrame()
        if column == "path":
            out["source_id"] = df[column].map(id_from_path)
        else:
            out["source_id"] = df[column].astype(str)
        out["source_id_norm"] = out["source_id"].map(normalize_id)
        return out.loc[out["source_id_norm"].ne("")].drop_duplicates("source_id_norm")
    raise ValueError(f"Could not read an ID column from provenance table: {path}")


def collect_secondary_provenance(
    brayden: pd.DataFrame,
    provenance_tables: Iterable[Path],
) -> pd.DataFrame:
    brayden_ids = set(brayden["source_id_norm"])
    source_by_norm = dict(zip(brayden["source_id_norm"], brayden["source_id"]))
    rows: list[dict[str, str]] = []

    for table in provenance_tables:
        path = Path(table)
        if not path.exists():
            continue
        try:
            ids = read_provenance_ids(path)
        except Exception as exc:
            rows.append(
                {
                    "source_id": "",
                    "provenance_table": str(path),
                    "provenance_status": f"read_error:{type(exc).__name__}",
                }
            )
            continue
        for source_id_norm in sorted(set(ids["source_id_norm"]) & brayden_ids):
            rows.append(
                {
                    "source_id": source_by_norm[source_id_norm],
                    "provenance_table": str(path),
                    "provenance_status": "present",
                }
            )

    return pd.DataFrame(rows, columns=["source_id", "provenance_table", "provenance_status"])


def build_audit(
    candidates: pd.DataFrame,
    reproduction: pd.DataFrame,
    *,
    brayden: pd.DataFrame | None = None,
    expected_brayden_count: int | None = 29,
    allow_unmatched_reproduction: bool = False,
    provenance_tables: Iterable[Path] | None = DEFAULT_PROVENANCE_TABLES,
) -> dict[str, object]:
    brayden_df = brayden.copy() if brayden is not None else brayden_candidates_frame()
    if expected_brayden_count is not None and len(brayden_df) != expected_brayden_count:
        raise ValueError(f"Expected {expected_brayden_count} Brayden targets, found {len(brayden_df)}")

    reproduction_unmatched = reproduction.loc[
        ~reproduction["source_id_norm"].isin(set(brayden_df["source_id_norm"]))
    ].copy()
    if not reproduction_unmatched.empty and not allow_unmatched_reproduction:
        sample = ", ".join(reproduction_unmatched["source_id"].astype(str).head(10).tolist())
        raise ValueError(f"Reproduction CSV has rows outside Brayden candidates: {sample}")

    candidate_columns = [
        column
        for column in ["asas_sn_id", "asas_sn_id_norm", "mag_bin", "path", "ra_deg", "dec_deg"]
        if column in candidates.columns
    ]
    candidate_slim = candidates[candidate_columns].rename(
        columns={
            "asas_sn_id": "v2_asas_sn_id",
            "mag_bin": "v2_mag_bin",
            "path": "v2_path",
            "ra_deg": "v2_ra_deg",
            "dec_deg": "v2_dec_deg",
        }
    )
    comparison = brayden_df.merge(
        candidate_slim,
        left_on="source_id_norm",
        right_on="asas_sn_id_norm",
        how="left",
        validate="one_to_one",
    ).drop(columns=["asas_sn_id_norm"])

    repro_keep = ["source_id_norm", "reproduction_detected"]
    repro_keep.extend(
        column for column in ["rejection_reason", "detection_details"] if column in reproduction.columns
    )
    metric_cols = [
        "g_bayes_dip_significant",
        "v_bayes_dip_significant",
        "g_bayes_dip_bayes_factor",
        "v_bayes_dip_bayes_factor",
        "g_n_runs",
        "v_n_runs",
    ]
    repro_keep.extend([column for column in metric_cols if column in reproduction.columns])
    repro_slim = reproduction[repro_keep].rename(
        columns={
            "rejection_reason": "reproduction_rejection_reason",
            "detection_details": "reproduction_detection_details",
        }
    )
    comparison = comparison.merge(
        repro_slim,
        on="source_id_norm",
        how="left",
        validate="one_to_one",
    )
    if comparison["reproduction_detected"].isna().any():
        missing = comparison.loc[comparison["reproduction_detected"].isna(), "source_id"].tolist()
        sample = ", ".join(map(str, missing[:10]))
        raise ValueError(f"Missing reproduction results for Brayden targets: {sample}")

    comparison["in_12_15_v2"] = comparison["v2_asas_sn_id"].notna()
    comparison["discrepancy_category"] = comparison.apply(classify_discrepancy, axis=1)

    compact_columns = [
        "source",
        "source_id",
        "category",
        "expected_mag_bin",
        "v2_mag_bin",
        "expected_detected",
        "reproduction_detected",
        "reproduction_rejection_reason",
        "reproduction_detection_details",
        "in_12_15_v2",
        "discrepancy_category",
    ]
    compact_columns.extend([column for column in metric_cols if column in comparison.columns])
    comparison = comparison[[column for column in compact_columns if column in comparison.columns]]

    overlap = (
        comparison.groupby("expected_mag_bin", dropna=False)
        .agg(
            brayden_total=("source_id", "size"),
            present_in_12_15_v2=("in_12_15_v2", "sum"),
        )
        .reset_index()
        .sort_values("expected_mag_bin")
    )
    overlap["missing_from_12_15_v2"] = overlap["brayden_total"] - overlap["present_in_12_15_v2"]
    overlap["fraction_present"] = overlap["present_in_12_15_v2"] / overlap["brayden_total"]

    discrepancy_counts = (
        comparison["discrepancy_category"]
        .value_counts()
        .rename_axis("discrepancy_category")
        .reset_index(name="count")
        .sort_values(["discrepancy_category"])
        .reset_index(drop=True)
    )
    mag_bin_mismatches = comparison.loc[
        comparison["discrepancy_category"].eq("mag_bin_mismatch")
    ].copy()

    provenance = pd.DataFrame(columns=["source_id", "provenance_table", "provenance_status"])
    if provenance_tables is not None:
        provenance = collect_secondary_provenance(brayden_df, provenance_tables)

    summary = {
        "primary_candidates_csv_rows": int(len(candidates)),
        "primary_candidates_unique_ids": int(candidates["asas_sn_id_norm"].nunique()),
        "brayden_targets": int(len(brayden_df)),
        "reproduction_rows": int(len(reproduction)),
        "reproduction_matched_to_brayden": int(
            reproduction["source_id_norm"].isin(set(brayden_df["source_id_norm"])).sum()
        ),
        "reproduction_unmatched_rows": int(len(reproduction_unmatched)),
        "brayden_present_in_12_15_v2": int(comparison["in_12_15_v2"].sum()),
        "brayden_missing_from_12_15_v2": int((~comparison["in_12_15_v2"]).sum()),
        "mag_bin_mismatches": int(len(mag_bin_mismatches)),
    }

    return {
        "comparison": comparison,
        "candidate_counts_by_mag_bin": candidate_counts_by_mag_bin(candidates),
        "brayden_overlap_by_expected_mag_bin": overlap,
        "discrepancy_counts_by_category": discrepancy_counts,
        "mag_bin_mismatches": mag_bin_mismatches,
        "secondary_parquet_provenance": provenance,
        "reproduction_unmatched": reproduction_unmatched.drop(columns=["source_id_norm"], errors="ignore"),
        "summary": summary,
    }


def write_reports(audit: dict[str, object], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    table_names = [
        "comparison",
        "candidate_counts_by_mag_bin",
        "brayden_overlap_by_expected_mag_bin",
        "discrepancy_counts_by_category",
        "mag_bin_mismatches",
        "secondary_parquet_provenance",
        "reproduction_unmatched",
    ]
    for name in table_names:
        value = audit[name]
        assert isinstance(value, pd.DataFrame)
        path = output_dir / f"{name}.csv"
        value.to_csv(path, index=False)
        outputs[name] = path

    summary_path = output_dir / "summary.json"
    summary = audit["summary"]
    assert isinstance(summary, dict)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    outputs["summary"] = summary_path
    return outputs


def run_audit(
    *,
    candidates_csv: Path = DEFAULT_CANDIDATES_CSV,
    reproduction_csv: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    allow_unmatched_reproduction: bool = False,
    provenance_tables: Iterable[Path] | None = DEFAULT_PROVENANCE_TABLES,
) -> dict[str, object]:
    reproduction_path = reproduction_csv or find_latest_reproduction_csv()
    candidates = load_primary_candidates(candidates_csv)
    reproduction = load_reproduction_results(reproduction_path)
    audit = build_audit(
        candidates,
        reproduction,
        allow_unmatched_reproduction=allow_unmatched_reproduction,
        provenance_tables=provenance_tables,
    )
    outputs = write_reports(audit, output_dir)
    audit["outputs"] = outputs
    audit["reproduction_csv"] = reproduction_path
    audit["candidates_csv"] = candidates_csv
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare output/12-15mag_candidates_v2.csv against Brayden reproduction results.",
    )
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES_CSV)
    parser.add_argument(
        "--reproduction",
        type=Path,
        default=None,
        help="Reproduction CSV. Defaults to newest output/logs/reproduction/reproduction_*.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-unmatched-reproduction",
        action="store_true",
        help="Do not fail if the reproduction CSV includes IDs outside Brayden's list.",
    )
    parser.add_argument(
        "--skip-provenance",
        action="store_true",
        help="Skip secondary parquet provenance checks.",
    )
    parser.add_argument(
        "--provenance-table",
        type=Path,
        action="append",
        default=None,
        help="Optional secondary parquet table to check for Brayden ID presence. May be repeated.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    provenance_tables: Iterable[Path] | None
    if args.skip_provenance:
        provenance_tables = None
    else:
        provenance_tables = tuple(args.provenance_table) if args.provenance_table else DEFAULT_PROVENANCE_TABLES

    try:
        audit = run_audit(
            candidates_csv=args.candidates,
            reproduction_csv=args.reproduction,
            output_dir=args.output_dir,
            allow_unmatched_reproduction=bool(args.allow_unmatched_reproduction),
            provenance_tables=provenance_tables,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Candidates CSV: {audit['candidates_csv']}")
    print(f"Reproduction CSV: {audit['reproduction_csv']}")
    print(f"Output dir: {args.output_dir}")
    for key, value in audit["summary"].items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
