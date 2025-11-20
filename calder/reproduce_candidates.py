from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from lc_dips import naive_dip_finder


brayden_candidates: list[dict[str, object]] = [
    {"source": "J042214+152530", "source_id": "377957522430", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J202402+383938", "source_id": "42950993887", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J174328+343315", "source_id": "223339338105", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J080327-261620", "source_id": "601296043597", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": False},
    {"source": "J184916-473251", "source_id": "472447294641", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Known", "expected_detected": True},
    {"source": "J183153-284827", "source_id": "455267102087", "category": "Dippers", "mag_bin": "13.5_14", "search_method": "Known", "expected_detected": False},
    {"source": "J070519+061219", "source_id": "266288137752", "category": "Dippers", "mag_bin": "13.5_14", "search_method": "Known", "expected_detected": False},
    {"source": "J081523-385923", "source_id": "532576686103", "category": "Dippers", "mag_bin": "13.5_14", "search_method": "Known", "expected_detected": False},
    {"source": "J085816-430955", "source_id": "352187470767", "category": "Dippers", "mag_bin": "12_12.5", "search_method": "Known", "expected_detected": False},
    {"source": "J114712-621037", "source_id": "609886184506", "category": "Dippers", "mag_bin": "13_13.5", "search_method": "Known", "expected_detected": False},
    {"source": "J005437+644347", "source_id": "68720274411", "category": "Multiple Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Known", "expected_detected": True},
    {"source": "J062510-075341", "source_id": "377958261591", "category": "Multiple Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J124745-622756", "source_id": "515397118400", "category": "Multiple Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J175912-120956", "source_id": "326417831663", "category": "Multiple Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J181752-580749", "source_id": "644245387906", "category": "Multiple Eclipse Binaries", "mag_bin": "12_12.5", "search_method": "Known", "expected_detected": True},
    {"source": "J160757-574540", "source_id": "661425129485", "category": "Multiple Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": False},
    {"source": "J073924-272916", "source_id": "438086977939", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J074007-161608", "source_id": "360777377116", "category": "Single Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J094848-545959", "source_id": "635655234580", "category": "Single Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J162209-444247", "source_id": "412317159120", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J183606-314826", "source_id": "438086901547", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J205245-713514", "source_id": "463856535113", "category": "Single Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J212132+480140", "source_id": "120259184943", "category": "Single Eclipse Binaries", "mag_bin": "13_13.5", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J225702+562312", "source_id": "25770019815", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J190316-195739", "source_id": "515396514761", "category": "Single Eclipse Binaries", "mag_bin": "13.5_14", "search_method": "Pipeline", "expected_detected": True},
    {"source": "J175602+013135", "source_id": "231929175915", "category": "Single Eclipse Binaries", "mag_bin": "14_14.5", "search_method": "Known", "expected_detected": True},
    {"source": "J073234-200049", "source_id": "335007754417", "category": "Single Eclipse Binaries", "mag_bin": "14.5_15", "search_method": "Known", "expected_detected": True},
    {"source": "J223332+565552", "source_id": "60130040391", "category": "Single Eclipse Binaries", "mag_bin": "12.5_13", "search_method": "Known", "expected_detected": True},
    {"source": "J183210-173432", "source_id": "317827964025", "category": "Single Eclipse Binaries", "mag_bin": "12.5_13", "search_method": "Pipeline", "expected_detected": False},
]

skypatrol_lightcurve_files = [
    "calder/lc_data_skypatrol/231929175915-light-curves.csv",
    "calder/lc_data_skypatrol/266288137752-light-curves.csv",
    "calder/lc_data_skypatrol/352187470767-light-curves.csv",
    "calder/lc_data_skypatrol/438086977939-light-curves.csv",
    "calder/lc_data_skypatrol/455267102087-light-curves.csv",
    "calder/lc_data_skypatrol/463856535113-light-curves.csv",
    "calder/lc_data_skypatrol/532576686103-light-curves.csv",
    "calder/lc_data_skypatrol/609886184506-light-curves.csv",
]


def _load_manifest_df(manifest_path: Path | str) -> pd.DataFrame:
    path = Path(manifest_path).expanduser()
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "source_id" not in df.columns and "asas_sn_id" in df.columns:
        df = df.rename(columns={"asas_sn_id": "source_id"})
    df["source_id"] = df["source_id"].astype(str)
    return df


def _dataframe_from_candidates(data: Sequence[Mapping[str, object]] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(data or brayden_candidates).copy()
    df.rename(columns={"Source": "source", "Source ID": "source_id"}, inplace=True)
    df["source_id"] = df["source_id"].astype(str)
    return df


def _target_map(df: pd.DataFrame) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for mag_bin, chunk in df.groupby("mag_bin"):
        grouped[mag_bin] = set(chunk["source_id"].astype(str))
    return grouped


def _records_from_manifest(df: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    records: dict[str, list[dict[str, object]]] = {}
    for rec in df.to_dict("records"):
        source_id = str(rec.get("source_id"))
        mag_bin = rec.get("mag_bin")
        lc_dir = rec.get("lc_dir")
        mag_bin_str = str(mag_bin)
        lc_dir_str = str(lc_dir)
        dat_path = rec.get("dat_path") or str(Path(lc_dir_str) / f"{source_id}.dat")
        record = {
            "mag_bin": mag_bin_str,
            "index_num": rec.get("index_num"),
            "index_csv": rec.get("index_csv"),
            "lc_dir": lc_dir_str,
            "asas_sn_id": source_id,
            "dat_path": dat_path,
            "found": bool(rec.get("dat_exists", True)),
        }
        records.setdefault(mag_bin_str, []).append(record)

    return records


def _coerce_candidate_records(data) -> list[dict[str, object]]:
    if data is None:
        return list(brayden_candidates)

    if isinstance(data, pd.DataFrame):
        records = data.to_dict("records")
    else:
        records = list(data)

    if not records:
        return []

    first = records[0]
    if isinstance(first, Mapping):
        coerced: list[dict[str, object]] = []
        for rec in records:
            if not isinstance(rec, Mapping):
                continue
            new = dict(rec)
            source_id = str(new.get("source_id", new.get("Source_ID", ""))).strip()
            if not source_id:
                continue
            new["source_id"] = source_id
            new.setdefault("source", new.get("source", source_id))
            coerced.append(new)
        return coerced

    # assume list of file paths or source ids, try to match known candidates
    ids = []
    for entry in records:
        entry_str = str(entry)
        entry_path = Path(entry_str)
        stem = entry_path.stem
        source_id = stem.split("-")[0] if stem else entry_str
        ids.append(source_id)

    lookup = {c["source_id"]: c for c in brayden_candidates}
    coerced = []
    seen = set()
    for source_id in ids:
        if source_id in seen:
            continue
        seen.add(source_id)
        if source_id in lookup:
            coerced.append(lookup[source_id])
        else:
            coerced.append({"source": source_id, "source_id": source_id, "mag_bin": None})
    return coerced


def _resolve_candidates(spec: str | None):
    if spec is None:
        return list(brayden_candidates)

    env = globals()
    if spec in env:
        return _coerce_candidate_records(env[spec])

    cand_path = Path(spec)
    if cand_path.exists():
        if cand_path.suffix.lower() in {".csv", ".tsv"}:
            df = pd.read_csv(cand_path)
        elif cand_path.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(cand_path)
        else:
            raise SystemExit(f"Unsupported candidates file format: {cand_path}")
        return _coerce_candidate_records(df)

    raise SystemExit(f"Unknown candidates spec '{spec}'. Provide a built-in name or a valid file path.")


def build_reproduction_report(
    candidates: Sequence[Mapping[str, object]] | None = None,
    *,
    out_dir: Path | str = "./peak_results_repro",
    out_format: str = "csv",
    n_workers: int | None = None,
    chunk_size: int = 250000,
    metrics_baseline_func=None,
    metrics_dip_threshold: float = 0.3,
    extra_columns: Iterable[str] | None = None,
    manifest_path: Path | str | None = None,
    **baseline_kwargs,
) -> pd.DataFrame:
    manifest_df = _load_manifest_df(manifest_path) if manifest_path is not None else None

    baseline_candidates = candidates if candidates is not None else brayden_candidates
    df_targets = _dataframe_from_candidates(baseline_candidates)

    manifest_subset = None
    if manifest_df is not None:
        manifest_subset = manifest_df[manifest_df["source_id"].isin(df_targets["source_id"])].copy()
        if not manifest_subset.empty:
            cols = [
                col
                for col in ["source_id", "mag_bin", "lc_dir", "index_num", "index_csv", "dat_path", "dat_exists"]
                if col in manifest_subset.columns
            ]
            df_targets = df_targets.merge(manifest_subset[cols], on="source_id", how="left")
            if "mag_bin_x" in df_targets.columns and "mag_bin_y" in df_targets.columns:
                df_targets["mag_bin"] = df_targets["mag_bin_y"].fillna(df_targets["mag_bin_x"])
                df_targets = df_targets.drop(columns=["mag_bin_x", "mag_bin_y"])

    target_map = _target_map(df_targets)
    records_map = _records_from_manifest(manifest_subset) if manifest_subset is not None else None

    rows = naive_dip_finder(
        mag_bins=sorted(target_map),
        out_dir=out_dir,
        out_format=out_format,
        n_workers=n_workers,
        chunk_size=chunk_size,
        metrics_baseline_func=metrics_baseline_func,
        metrics_dip_threshold=metrics_dip_threshold,
        target_ids_by_bin=target_map,
        records_by_bin=records_map,
        return_rows=True,
        **baseline_kwargs,
    )

    rows_df = rows.copy() if rows is not None else pd.DataFrame()
    if rows_df.empty:
        rows_df = pd.DataFrame(columns=["source_id", "mag_bin"])
    else:
        rows_df = rows_df.copy()
    if "source_id" not in rows_df.columns:
        if "asas_sn_id" in rows_df.columns:
            rows_df["source_id"] = rows_df["asas_sn_id"].astype(str)
        else:
            rows_df["source_id"] = ""
    rows_df = rows_df.drop(columns=["asas_sn_id"], errors="ignore")
    if "mag_bin" not in rows_df.columns:
        rows_df["mag_bin"] = ""

    merged = df_targets.merge(rows_df, on=["source_id", "mag_bin"], how="left", suffixes=("", "_det"))

    g_peaks = merged["g_n_peaks"] if "g_n_peaks" in merged.columns else pd.Series(np.nan, index=merged.index)
    v_peaks = merged["v_n_peaks"] if "v_n_peaks" in merged.columns else pd.Series(np.nan, index=merged.index)
    merged["detected"] = (
        (g_peaks.fillna(0).astype(float) > 0) | (v_peaks.fillna(0).astype(float) > 0)
    )
    merged["detection_details"] = merged.apply(
        lambda row: (
            f"mag_bin={row['mag_bin']}; g_peaks={int(row['g_n_peaks'])}; v_peaks={int(row['v_n_peaks'])}"
            if row["detected"]
            else "—"
        ),
        axis=1,
    )
    if "expected_detected" in merged.columns:
        merged["matches_expected"] = merged["detected"] == merged["expected_detected"].astype(bool)

    if extra_columns:
        cols = [c for c in extra_columns if c in merged.columns]
    else:
        cols = []

    ordered_cols = [
        "source",
        "source_id",
        "category",
        "mag_bin",
        "detected",
        "detection_details",
    ]
    for col in [
        "expected_detected",
        "matches_expected",
        "g_n_peaks",
        "v_n_peaks",
        "g_max_depth",
        "v_max_depth",
        "jd_first",
        "jd_last",
    ]:
        if col in merged.columns:
            ordered_cols.append(col)
    ordered_cols.extend([c for c in cols if c not in ordered_cols])

    remaining = [c for c in merged.columns if c not in ordered_cols]
    ordered_cols.extend(remaining)

    return merged[ordered_cols]


__all__ = [
    "brayden_candidates",
    "skypatrol_lightcurve_files",
    "build_reproduction_report",
]

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the targeted reproduction search and summarize results."
    )
    parser.add_argument("--out-dir", default="./results_test", help="Directory for peak_results output")
    parser.add_argument("--out-format", choices=("csv", "parquet"), default="csv")
    parser.add_argument("--n-workers", type=int, default=10, help="ProcessPool worker count for naive_dip_finder")
    parser.add_argument("--chunk-size", type=int, default=250000, help="Rows per chunk flush for CSV output")
    parser.add_argument("--metrics-dip-threshold", type=float, default=0.3, help="Dip threshold for run_metrics")
    parser.add_argument("--manifest", default=None, help="Path to lc_manifest CSV/Parquet for targeted reproduction")
    parser.add_argument("--baseline-func", default=None, help="Baseline function import path (e.g. module:func)")
    parser.add_argument("--candidates", default=None, help="Candidate spec (built-in list name or path to CSV/Parquet file).",
    )
    return parser

def main(argv: Iterable[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    kwargs = {}
    if args.baseline_func:
        if ":" in args.baseline_func:
            mod_name, func_name = args.baseline_func.split(":", 1)
        else:
            mod_name, func_name = "lc_baseline", args.baseline_func
        mod = __import__(mod_name, fromlist=[func_name])
        kwargs["baseline_func"] = getattr(mod, func_name)

    candidate_data = _resolve_candidates(args.candidates)

    report = build_reproduction_report(
        candidates=candidate_data,
        out_dir=args.out_dir,
        out_format=args.out_format,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
        metrics_dip_threshold=args.metrics_dip_threshold,
        manifest_path=args.manifest,
        **kwargs,
    )

    columns = [
        "source",
        "source_id",
        "mag_bin",
        "detected",
        "detection_details",
        "g_n_peaks",
        "v_n_peaks",
        "matches_expected",
    ]
    existing = [c for c in columns if c in report.columns]
    print(report[existing].to_string(index=False))


if __name__ == "__main__":
    main()