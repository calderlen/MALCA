from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sqlite3
import subprocess
from typing import Any, Iterable

import pandas as pd

from malca.config import DEFAULT_OUTPUT_DIR
from malca.products.feature_layers import with_feature_columns
from malca.ltv.paths import DEFAULT_LTV_RUN_DIR, discover_ltv_output_dir, default_ltv_review_db_for_output
from malca.io.table_io import read_feature_table


def _read_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path).expanduser()
    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(table_path)
    if suffix == ".json":
        return pd.read_json(table_path)
    return read_feature_table(table_path)


def _read_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path).expanduser()
    if not config_path.exists():
        return {"__error__": f"missing: {config_path}"}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"__error__": str(exc)}
    return data if isinstance(data, dict) else {"__value__": data}


def _key_series(df: pd.DataFrame, key: str) -> pd.Series:
    df = with_feature_columns(df, [key])
    if key in df.columns:
        return df[key].fillna("").astype(str)
    for fallback in ("lc_path", "candidate_id", "asas_sn_id", "source_id", "gaia_id"):
        df = with_feature_columns(df, [fallback])
        if fallback in df.columns:
            return df[fallback].fillna("").astype(str)
    raise ValueError(f"Key column {key!r} was not found and no fallback ID column is available.")


def _table_summary(df: pd.DataFrame, key: str) -> dict[str, Any]:
    df = with_feature_columns(df, ["failed_any", "filter_reason"])
    keys = _key_series(df, key)
    nonempty = keys[keys != ""]
    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "unique_keys": int(nonempty.nunique(dropna=True)),
        "duplicate_keys": int(nonempty.duplicated().sum()),
        "columns": list(map(str, df.columns)),
    }
    if "failed_any" in df.columns:
        failed = df["failed_any"]
        if pd.api.types.is_bool_dtype(failed):
            summary["failed_any_true"] = int(failed.fillna(False).sum())
        else:
            summary["failed_any_true"] = int(failed.fillna(0).astype(float).ne(0).sum())
    if "filter_reason" in df.columns:
        summary["filter_reason_counts"] = {
            str(k): int(v)
            for k, v in df["filter_reason"].fillna("").astype(str).value_counts(dropna=False).head(25).items()
        }
    return summary


def _set_diff(left: pd.DataFrame, right: pd.DataFrame, key: str, sample: int) -> dict[str, Any]:
    left_keys = set(_key_series(left, key))
    right_keys = set(_key_series(right, key))
    left_keys.discard("")
    right_keys.discard("")
    only_left = sorted(left_keys - right_keys)
    only_right = sorted(right_keys - left_keys)
    return {
        "left_only_count": int(len(only_left)),
        "right_only_count": int(len(only_right)),
        "left_only_sample": only_left[:sample],
        "right_only_sample": only_right[:sample],
    }


def _config_diff(left: dict[str, Any], right: dict[str, Any], sample: int) -> dict[str, Any]:
    keys = sorted(set(left) | set(right))
    changed = []
    for key in keys:
        if left.get(key) != right.get(key):
            changed.append({"key": key, "left": left.get(key), "right": right.get(key)})
    return {
        "changed_count": int(len(changed)),
        "changed_sample": changed[:sample],
    }


def compare_results(
    raw: str | Path,
    filtered: str | Path,
    vetted: str | Path | None = None,
    *,
    key: str = "path",
    raw_config: str | Path | None = None,
    filtered_config: str | Path | None = None,
    vetted_config: str | Path | None = None,
    sample: int = 20,
) -> dict[str, Any]:
    raw_df = _read_table(raw)
    filtered_df = _read_table(filtered)
    report: dict[str, Any] = {
        "raw": _table_summary(raw_df, key),
        "filtered": _table_summary(filtered_df, key),
        "raw_vs_filtered": _set_diff(raw_df, filtered_df, key, sample),
    }
    if vetted is not None:
        vetted_df = _read_table(vetted)
        report["vetted"] = _table_summary(vetted_df, key)
        report["filtered_vs_vetted"] = _set_diff(filtered_df, vetted_df, key, sample)
        report["raw_vs_vetted"] = _set_diff(raw_df, vetted_df, key, sample)

    raw_cfg = _read_config(raw_config)
    filtered_cfg = _read_config(filtered_config)
    vetted_cfg = _read_config(vetted_config)
    if raw_cfg or filtered_cfg:
        report["raw_vs_filtered_config"] = _config_diff(raw_cfg, filtered_cfg, sample)
    if filtered_cfg or vetted_cfg:
        report["filtered_vs_vetted_config"] = _config_diff(filtered_cfg, vetted_cfg, sample)
    return report


def _parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    if path.is_dir():
        return int(sum(_parquet_rows(child) for child in sorted(path.glob("*.parquet"))))
    metadata = pq.ParquetFile(path).metadata
    return int(metadata.num_rows)


def _processed_line_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for line in handle if line.strip())
    except Exception:
        return None


def _review_db_summary(review_db: str | Path | None) -> dict[str, Any]:
    if review_db is None:
        return {}
    db_path = Path(review_db).expanduser()
    if not db_path.exists():
        return {"path": str(db_path), "exists": False}
    out: dict[str, Any] = {"path": str(db_path), "exists": True}
    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "candidates" in tables:
            out["candidate_rows"] = int(conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
        if "reviews" in tables:
            out["review_rows"] = int(conn.execute("SELECT COUNT(*) FROM reviews").fetchone()[0])
            out["review_status_counts"] = {
                str(status): int(count)
                for status, count in conn.execute(
                    "SELECT coalesce(status, ''), COUNT(*) FROM reviews GROUP BY coalesce(status, '')"
                ).fetchall()
            }
            out["event_class_counts"] = {
                str(label): int(count)
                for label, count in conn.execute(
                    "SELECT coalesce(event_class, ''), COUNT(*) FROM reviews GROUP BY coalesce(event_class, '')"
                ).fetchall()
            }
    return out


def ltv_status(
    output_dir: str | Path | None = None,
    *,
    review_db: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(output_dir).expanduser() if output_dir is not None else discover_ltv_output_dir()
    resolved_review_db = review_db if review_db is not None else default_ltv_review_db_for_output(root)
    bins = []
    if root.exists():
        candidates = sorted(root.glob("LTvar*.parquet"))
        candidates.extend(sorted(root.glob("LTvar*_pipeline.parquet")))
        seen: set[str] = set()
        for path in candidates:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            record: dict[str, Any] = {
                "path": str(path),
                "name": path.name,
                "exists": path.exists(),
                "is_directory_dataset": path.is_dir(),
            }
            try:
                record["rows"] = _parquet_rows(path)
            except Exception as exc:
                record["rows_error"] = str(exc)
            if path.name.endswith("_pipeline.parquet"):
                processed_path = path.with_name(path.name.replace("_pipeline.parquet", "_PROCESSED.txt"))
            else:
                processed_path = path.with_name(path.name.replace(".parquet", "_PROCESSED.txt"))
            processed_count = _processed_line_count(processed_path)
            record["processed_path"] = str(processed_path)
            record["processed_lines"] = processed_count
            if processed_count is not None and isinstance(record.get("rows"), int):
                record["processed_minus_rows"] = int(processed_count - int(record["rows"]))
            bins.append(record)
    return {
        "output_dir": str(root),
        "exists": root.exists(),
        "bins": bins,
        "review_db": _review_db_summary(resolved_review_db),
    }


def _write_smoke_candidates(candidate_list: Path, output_root: Path, smoke_count: int) -> Path:
    df = _read_table(candidate_list)
    smoke = df.head(max(int(smoke_count), 1)).copy()
    output_root.mkdir(parents=True, exist_ok=True)
    smoke_path = output_root / f"{candidate_list.stem}_smoke_{len(smoke)}.csv"
    smoke.to_csv(smoke_path, index=False)
    return smoke_path


def baseline_compare_commands(
    candidate_list: str | Path = "output/lc_events_collect_candidates_14_14.5.csv",
    *,
    output_root: str | Path = DEFAULT_OUTPUT_DIR / "audit" / "baseline_compare",
    smoke_count: int = 100,
    workers: int = 1,
    path_prefix: str | None = None,
    path_root: str | None = None,
    extra_reproduce_args: Iterable[str] = (),
) -> dict[str, Any]:
    candidates = Path(candidate_list).expanduser()
    out_root = Path(output_root).expanduser()
    smoke_candidates = _write_smoke_candidates(candidates, out_root, smoke_count)
    common = [
        "malca",
        "reproduce",
        "--workers",
        str(int(workers)),
        "--log-format",
        "parquet",
    ]
    if path_prefix:
        common.extend(["--path-prefix", str(path_prefix)])
    if path_root:
        common.extend(["--path-root", str(path_root)])
    common.extend(map(str, extra_reproduce_args))

    smoke_cmd = [
        *common,
        "--candidates",
        str(smoke_candidates),
        "--output-dir",
        str(out_root / "global_median_smoke"),
        "--baseline-func",
        "global_median",
    ]
    full_cmd = [
        *common,
        "--candidates",
        str(candidates),
        "--output-dir",
        str(out_root / "per_camera_median_full"),
        "--baseline-func",
        "per_camera_median",
    ]
    return {
        "smoke_candidates": str(smoke_candidates),
        "smoke_command": smoke_cmd,
        "full_command": full_cmd,
        "smoke_command_text": shlex.join(smoke_cmd),
        "full_command_text": shlex.join(full_cmd),
    }


def _print_json(data: dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, default=str))


def _run_compare_results(args: argparse.Namespace) -> None:
    _print_json(
        compare_results(
            args.raw,
            args.filtered,
            args.vetted,
            key=args.key,
            raw_config=args.raw_config,
            filtered_config=args.filtered_config,
            vetted_config=args.vetted_config,
            sample=args.sample,
        )
    )


def _run_ltv_status(args: argparse.Namespace) -> None:
    _print_json(ltv_status(args.output_dir, review_db=args.review_db))


def _run_baseline_compare(args: argparse.Namespace) -> None:
    report = baseline_compare_commands(
        args.candidate_list,
        output_root=args.output_root,
        smoke_count=args.smoke_count,
        workers=args.workers,
        path_prefix=args.path_prefix,
        path_root=args.path_root,
        extra_reproduce_args=args.extra_reproduce_arg,
    )
    if args.execute:
        subprocess.run(report["smoke_command"], check=True)
        subprocess.run(report["full_command"], check=True)
    _print_json(report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MALCA audit helpers for result, LTV, and baseline comparisons.")
    sub = parser.add_subparsers(dest="command", required=True)

    compare = sub.add_parser("compare-results", help="Compare raw, filtered, and optionally vetted result tables.")
    compare.add_argument("--raw", required=True, help="Raw result table path.")
    compare.add_argument("--filtered", required=True, help="Filtered result table path.")
    compare.add_argument("--vetted", default=None, help="Optional vetted result table path.")
    compare.add_argument("--key", default="path", help="Primary key column for set comparisons.")
    compare.add_argument("--raw-config", default=None, help="Optional raw run_params/config JSON.")
    compare.add_argument("--filtered-config", default=None, help="Optional filtered run_params/config JSON.")
    compare.add_argument("--vetted-config", default=None, help="Optional vetted run_params/config JSON.")
    compare.add_argument("--sample", type=int, default=20, help="Number of sample keys/config diffs to include.")
    compare.set_defaults(func=_run_compare_results)

    ltv = sub.add_parser("ltv-status", help="Summarize LTV chunk outputs and review completion.")
    ltv.add_argument(
        "--output-dir",
        default=None,
        help=f"Directory containing LTV parquet datasets (default: discover {DEFAULT_LTV_RUN_DIR}, then {DEFAULT_OUTPUT_DIR / 'runs'}/ltv_*, then legacy output/ltv/ltv).",
    )
    ltv.add_argument(
        "--review-db",
        default=None,
        help="Review DB to summarize (default: <run>/review/review.db when using run-style output).",
    )
    ltv.set_defaults(func=_run_ltv_status)

    baseline = sub.add_parser("baseline-compare", help="Build or run global-median smoke and per-camera full commands.")
    baseline.add_argument("--candidate-list", default="output/lc_events_collect_candidates_14_14.5.csv")
    baseline.add_argument("--output-root", default=str(DEFAULT_OUTPUT_DIR / "audit" / "baseline_compare"))
    baseline.add_argument("--smoke-count", type=int, default=100)
    baseline.add_argument("--workers", type=int, default=1)
    baseline.add_argument("--path-prefix", default=None)
    baseline.add_argument("--path-root", default=None)
    baseline.add_argument("--extra-reproduce-arg", action="append", default=[])
    baseline.add_argument("--execute", action="store_true", help="Run smoke first, then the full per-camera command.")
    baseline.set_defaults(func=_run_baseline_compare)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
