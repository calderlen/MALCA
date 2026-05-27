#!/usr/bin/env python3
"""Audit candidates whose RA/Dec are missing from a candidate CSV.

This script is meant to run either on the cluster, where the absolute light-curve
paths in the candidate CSV exist, or on a local machine with bundled light curves.
It answers a narrow question:

    Are missing candidate coordinates absent from the source metadata too, or were
    they lost in an intermediate/export step?

It checks, in order:
  1. the candidate CSV rows with missing ra_deg/dec_deg,
  2. the resolved light-curve file path,
  3. the corresponding lcsv2_masked indexN_masked.csv when available, and
  4. an optional ASAS-SN index parquet/CSV.

The ASAS-SN .dat/.dat2/.dat3 files normally do not carry RA/Dec. For those files,
the masked index and ASAS-SN index are the meaningful coordinate sources.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_CANDIDATES = Path("output/12-15mag_candidates_v2_minus_previous.csv")
DEFAULT_OUTPUT = Path("output/missing_coord_audit.csv")
DEFAULT_SUMMARY = Path("output/missing_coord_audit_summary.json")
DEFAULT_ASASSN_INDEX_CANDIDATES = (
    Path("output/asassn_index_masked_concat_cleaned_20250919_154524_brotli.parquet"),
    Path("input/asassn_index_masked_concat_cleaned_20250919_154524_brotli.parquet"),
)
DEFAULT_LIGHTCURVE_ROOT = Path("/data/poohbah/1/assassin/rowan.90/lcsv2")
DEFAULT_MASKED_INDEX_ROOT = Path("/data/poohbah/1/assassin/lenhart/malca-older/calder/lcsv2_masked")

ID_COLS = ("asas_sn_id", "candidate_id", "source_id", "id")
RA_COLS = ("ra_deg", "ra", "raj2000", "ra_j2000", "ra_icrs", "RA", "RAJ2000")
DEC_COLS = ("dec_deg", "dec", "dej2000", "dec_j2000", "dec_icrs", "DEC", "DEJ2000")
LC_PATH_RE = re.compile(
    r"/(?P<mag_bin>\d+(?:\.\d+)?_\d+(?:\.\d+)?)/(?P<lc_dir>lc(?P<index_num>\d+)_cal)/",
)
INTEGER_FLOAT_RE = re.compile(r"^([+-]?\d+)\.0+$")
HEADER_COORD_RE = re.compile(
    r"\b(?P<label>ra_deg|dec_deg|raj2000|dej2000|ra|dec)\b\s*[:=,]\s*"
    r"(?P<value>[+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


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


def is_missing_coord(series: pd.Series) -> pd.Series:
    ra_missing = series["ra_deg"].isna() | (series["ra_deg"].astype(str).str.strip() == "")
    dec_missing = series["dec_deg"].isna() | (series["dec_deg"].astype(str).str.strip() == "")
    return ra_missing | dec_missing


def choose_column(columns: Iterable[str], choices: Iterable[str]) -> str | None:
    exact = {str(c): str(c) for c in columns}
    lower = {str(c).lower(): str(c) for c in columns}
    for choice in choices:
        if choice in exact:
            return exact[choice]
        if choice.lower() in lower:
            return lower[choice.lower()]
    return None


def choose_id_column(df: pd.DataFrame, explicit: str | None = None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Missing requested ID column {explicit!r}; columns: {', '.join(map(str, df.columns))}")
        return explicit
    col = choose_column(df.columns, ID_COLS)
    if col:
        return col
    raise ValueError(f"Could not find an ID column. Expected one of {ID_COLS}.")


def read_table(path: Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False, usecols=columns)
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    raise ValueError(f"Unsupported table type: {path}")


def infer_path_parts(row: pd.Series) -> dict[str, str]:
    path_text = finite_text(row.get("path"))
    match = LC_PATH_RE.search(path_text)
    parts = {
        "path_mag_bin": "",
        "path_lc_dir": "",
        "path_index_num": "",
    }
    if match:
        parts.update(
            {
                "path_mag_bin": match.group("mag_bin"),
                "path_lc_dir": match.group("lc_dir"),
                "path_index_num": match.group("index_num"),
            },
        )
    if not parts["path_mag_bin"]:
        parts["path_mag_bin"] = finite_text(row.get("mag_bin"))
    return parts


def candidate_lightcurve_paths(
    row: pd.Series,
    *,
    lightcurve_root: Path | None,
    bundle_lightcurve_dir: Path | None,
    extensions: tuple[str, ...],
) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()

    def add(path: Path | str | None) -> None:
        if path is None:
            return
        p = Path(str(path)).expanduser()
        key = str(p)
        if key not in seen:
            seen.add(key)
            paths.append(p)

    source_path = finite_text(row.get("path"))
    if source_path:
        add(source_path)

    candidate_id = finite_text(row.get("asas_sn_id")) or finite_text(row.get("candidate_id"))
    parts = infer_path_parts(row)
    suffixes = []
    if source_path:
        suffix = Path(source_path).suffix.lstrip(".")
        if suffix:
            suffixes.append(suffix)
    suffixes.extend(ext for ext in extensions if ext not in suffixes)

    if candidate_id and bundle_lightcurve_dir:
        for ext in suffixes:
            add(bundle_lightcurve_dir / f"{candidate_id}.{ext}")

    if candidate_id and lightcurve_root and parts["path_mag_bin"] and parts["path_lc_dir"]:
        for ext in suffixes:
            add(lightcurve_root / parts["path_mag_bin"] / parts["path_lc_dir"] / f"{candidate_id}.{ext}")

    return paths


def resolve_lightcurve_path(
    row: pd.Series,
    *,
    lightcurve_root: Path | None,
    bundle_lightcurve_dir: Path | None,
    extensions: tuple[str, ...],
) -> tuple[Path | None, list[str]]:
    candidates = candidate_lightcurve_paths(
        row,
        lightcurve_root=lightcurve_root,
        bundle_lightcurve_dir=bundle_lightcurve_dir,
        extensions=extensions,
    )
    for path in candidates:
        try:
            if path.exists():
                return path, [str(p) for p in candidates]
        except OSError:
            continue
    return None, [str(p) for p in candidates]


def parse_coords_from_lightcurve_header(path: Path) -> tuple[str, str, str]:
    suffix = path.suffix.lower()
    if suffix in {".dat", ".dat2", ".dat3", ".raw", ".raw2"}:
        return "", "", f"not_in_{suffix.lstrip('.')}_schema"

    try:
        text = path.read_text(errors="replace")[:32768]
    except Exception as exc:
        return "", "", f"read_error:{type(exc).__name__}"

    found: dict[str, str] = {}
    for match in HEADER_COORD_RE.finditer(text):
        found[match.group("label").lower()] = match.group("value")

    ra = next((found.get(c.lower()) for c in RA_COLS if found.get(c.lower())), "")
    dec = next((found.get(c.lower()) for c in DEC_COLS if found.get(c.lower())), "")
    if ra and dec:
        return ra, dec, "header_coords_found"
    return ra or "", dec or "", "no_header_coords_found"


@dataclass
class CoordinateLookup:
    rows: dict[str, tuple[str, str]]
    source_label: str

    def get(self, candidate_id: object) -> tuple[str, str]:
        return self.rows.get(normalize_id(candidate_id), ("", ""))


def build_asassn_index_lookup(index_path: Path | None, wanted_ids: set[str]) -> CoordinateLookup:
    if index_path is None or not index_path.exists():
        return CoordinateLookup({}, "not_checked")

    suffix = index_path.suffix.lower()
    if suffix == ".csv":
        header = pd.read_csv(index_path, nrows=0)
        id_col = choose_column(header.columns, ID_COLS)
        ra_col = choose_column(header.columns, RA_COLS)
        dec_col = choose_column(header.columns, DEC_COLS)
        if not id_col or not ra_col or not dec_col:
            return CoordinateLookup({}, f"missing_columns:{index_path}")
        out: dict[str, tuple[str, str]] = {}
        for chunk in pd.read_csv(index_path, dtype=str, usecols=[id_col, ra_col, dec_col], chunksize=250_000):
            chunk["_id_norm"] = chunk[id_col].map(normalize_id)
            hit = chunk[chunk["_id_norm"].isin(wanted_ids)]
            for row in hit.itertuples(index=False):
                data = row._asdict()
                ra = finite_text(data.get(ra_col))
                dec = finite_text(data.get(dec_col))
                if ra and dec:
                    out[data["_id_norm"]] = (ra, dec)
        return CoordinateLookup(out, str(index_path))

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to scan parquet ASAS-SN indexes") from exc

        pf = pq.ParquetFile(index_path)
        names = pf.schema.names
        id_col = choose_column(names, ID_COLS)
        ra_col = choose_column(names, RA_COLS)
        dec_col = choose_column(names, DEC_COLS)
        if not id_col or not ra_col or not dec_col:
            return CoordinateLookup({}, f"missing_columns:{index_path}")

        out: dict[str, tuple[str, str]] = {}
        for group_idx in range(pf.num_row_groups):
            chunk = pf.read_row_group(group_idx, columns=[id_col, ra_col, dec_col]).to_pandas()
            chunk["_id_norm"] = chunk[id_col].map(normalize_id)
            hit = chunk[chunk["_id_norm"].isin(wanted_ids)]
            for _, row in hit.iterrows():
                ra = finite_text(row.get(ra_col))
                dec = finite_text(row.get(dec_col))
                if ra and dec:
                    out[str(row["_id_norm"])] = (ra, dec)
        return CoordinateLookup(out, str(index_path))

    return CoordinateLookup({}, f"unsupported:{index_path}")


class MaskedIndexLookup:
    def __init__(self, root: Path | None, id_col: str | None = None) -> None:
        self.root = root
        self.id_col = id_col
        self._cache: dict[Path, tuple[str, dict[str, tuple[str, str]]]] = {}

    def index_path_for_row(self, row: pd.Series) -> Path | None:
        if self.root is None:
            return None
        parts = infer_path_parts(row)
        if not parts["path_mag_bin"] or not parts["path_index_num"]:
            return None
        return self.root / parts["path_mag_bin"] / f"index{parts['path_index_num']}_masked.csv"

    def load(self, index_path: Path) -> tuple[str, dict[str, tuple[str, str]]]:
        if index_path in self._cache:
            return self._cache[index_path]
        if not index_path.exists():
            result = ("missing", {})
            self._cache[index_path] = result
            return result

        try:
            header = pd.read_csv(index_path, nrows=0)
            id_col = choose_column(header.columns, [self.id_col] if self.id_col else ID_COLS)
            ra_col = choose_column(header.columns, RA_COLS)
            dec_col = choose_column(header.columns, DEC_COLS)
            if not id_col or not ra_col or not dec_col:
                result = ("missing_coordinate_columns", {})
                self._cache[index_path] = result
                return result

            df = pd.read_csv(index_path, dtype=str, usecols=[id_col, ra_col, dec_col], keep_default_na=False)
            df["_id_norm"] = df[id_col].map(normalize_id)
            lookup: dict[str, tuple[str, str]] = {}
            for _, row in df.iterrows():
                ra = finite_text(row.get(ra_col))
                dec = finite_text(row.get(dec_col))
                if ra and dec:
                    lookup[str(row["_id_norm"])] = (ra, dec)
            result = ("loaded", lookup)
            self._cache[index_path] = result
            return result
        except Exception as exc:
            result = (f"read_error:{type(exc).__name__}", {})
            self._cache[index_path] = result
            return result

    def get(self, row: pd.Series) -> tuple[str, str, str, str]:
        index_path = self.index_path_for_row(row)
        if index_path is None:
            return "", "", "", "not_resolved"
        status, lookup = self.load(index_path)
        ra, dec = lookup.get(normalize_id(row.get("asas_sn_id")), ("", ""))
        return str(index_path), ra, dec, status


def classify_diagnosis(
    *,
    lc_exists: bool,
    lc_ra: str,
    lc_dec: str,
    masked_ra: str,
    masked_dec: str,
    asassn_ra: str,
    asassn_dec: str,
) -> str:
    if lc_ra and lc_dec:
        return "lightcurve_has_coords_csv_missing"
    masked_has = bool(masked_ra and masked_dec)
    asassn_has = bool(asassn_ra and asassn_dec)
    if masked_has and asassn_has:
        return "csv_missing_but_masked_and_asassn_indexes_have_coords"
    if masked_has:
        return "csv_missing_but_masked_index_has_coords"
    if asassn_has:
        return "csv_missing_but_asassn_index_has_coords"
    if not lc_exists:
        return "lightcurve_path_missing_or_unmounted_and_no_checked_metadata_coords"
    return "coords_absent_from_checked_sources"


def resolve_default_asassn_index(explicit: Path | None) -> Path | None:
    if explicit:
        return explicit.expanduser()
    for candidate in DEFAULT_ASASSN_INDEX_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit missing RA/Dec rows in a candidate CSV against LC paths and metadata indexes.",
    )
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--id-column", default=None, help="Candidate CSV ID column. Default: auto-detect.")
    parser.add_argument("--asassn-index", type=Path, default=None, help="Optional ASAS-SN index parquet/CSV.")
    parser.add_argument(
        "--no-asassn-index",
        action="store_true",
        help="Skip ASAS-SN index lookup even if a default index exists.",
    )
    parser.add_argument(
        "--lightcurve-root",
        type=Path,
        default=DEFAULT_LIGHTCURVE_ROOT,
        help="Cluster LCV2 root used to resolve paths if the CSV path is not directly accessible.",
    )
    parser.add_argument(
        "--masked-index-root",
        type=Path,
        default=DEFAULT_MASKED_INDEX_ROOT,
        help="Cluster lcsv2_masked root containing <mag_bin>/indexN_masked.csv files.",
    )
    parser.add_argument(
        "--bundle-lightcurve-dir",
        type=Path,
        default=None,
        help="Optional local bundle_assets/lightcurves directory for local audits.",
    )
    parser.add_argument(
        "--extensions",
        default="dat3,dat2,dat,raw2,csv",
        help="Extensions to try when resolving by ID. Default: dat3,dat2,dat,raw2,csv.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only audit the first N missing-coordinate candidates. Default: all.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    candidates_path = args.candidates.expanduser()
    if not candidates_path.exists():
        print(f"Missing candidate CSV: {candidates_path}", file=sys.stderr)
        return 2

    candidates = pd.read_csv(candidates_path, dtype=str)
    id_col = choose_id_column(candidates, args.id_column)
    if id_col != "asas_sn_id":
        candidates["asas_sn_id"] = candidates[id_col]
    for col in ("ra_deg", "dec_deg"):
        if col not in candidates.columns:
            candidates[col] = ""

    missing = candidates[is_missing_coord(candidates)].copy()
    if args.limit and args.limit > 0:
        missing = missing.head(args.limit).copy()

    missing["_id_norm"] = missing["asas_sn_id"].map(normalize_id)
    wanted_ids = {x for x in missing["_id_norm"] if x}

    asassn_index_path = None if args.no_asassn_index else resolve_default_asassn_index(args.asassn_index)
    asassn_lookup = build_asassn_index_lookup(asassn_index_path, wanted_ids)

    masked_lookup = MaskedIndexLookup(args.masked_index_root.expanduser() if args.masked_index_root else None)
    extensions = tuple(ext.strip().lstrip(".") for ext in args.extensions.split(",") if ext.strip())

    rows: list[dict[str, object]] = []
    for _, row in missing.iterrows():
        parts = infer_path_parts(row)
        lc_path, attempted_paths = resolve_lightcurve_path(
            row,
            lightcurve_root=args.lightcurve_root.expanduser() if args.lightcurve_root else None,
            bundle_lightcurve_dir=args.bundle_lightcurve_dir.expanduser() if args.bundle_lightcurve_dir else None,
            extensions=extensions,
        )
        lc_exists = lc_path is not None
        lc_ra = lc_dec = ""
        lc_coord_status = "not_checked"
        lc_size_bytes = ""
        if lc_path is not None:
            try:
                lc_size_bytes = lc_path.stat().st_size
            except OSError:
                lc_size_bytes = ""
            lc_ra, lc_dec, lc_coord_status = parse_coords_from_lightcurve_header(lc_path)

        masked_index_path, masked_ra, masked_dec, masked_status = masked_lookup.get(row)
        asassn_ra, asassn_dec = asassn_lookup.get(row.get("asas_sn_id"))
        diagnosis = classify_diagnosis(
            lc_exists=lc_exists,
            lc_ra=lc_ra,
            lc_dec=lc_dec,
            masked_ra=masked_ra,
            masked_dec=masked_dec,
            asassn_ra=asassn_ra,
            asassn_dec=asassn_dec,
        )

        rows.append(
            {
                "asas_sn_id": row.get("asas_sn_id"),
                "mag_bin": row.get("mag_bin"),
                "candidate_csv_path": row.get("path"),
                "path_mag_bin": parts["path_mag_bin"],
                "path_lc_dir": parts["path_lc_dir"],
                "path_index_num": parts["path_index_num"],
                "resolved_lc_path": str(lc_path) if lc_path else "",
                "lc_exists": lc_exists,
                "lc_size_bytes": lc_size_bytes,
                "lc_coord_status": lc_coord_status,
                "lc_ra_deg": lc_ra,
                "lc_dec_deg": lc_dec,
                "masked_index_path": masked_index_path,
                "masked_index_status": masked_status,
                "masked_index_ra_deg": masked_ra,
                "masked_index_dec_deg": masked_dec,
                "asassn_index_source": asassn_lookup.source_label,
                "asassn_index_ra_deg": asassn_ra,
                "asassn_index_dec_deg": asassn_dec,
                "diagnosis": diagnosis,
                "attempted_lc_paths": " | ".join(attempted_paths),
            },
        )

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    summary = {
        "candidate_csv": str(candidates_path),
        "candidate_rows": int(len(candidates)),
        "audited_missing_coord_rows": int(len(missing)),
        "output_csv": str(args.output),
        "summary_json": str(args.summary),
        "asassn_index_source": asassn_lookup.source_label,
        "diagnosis_counts": out["diagnosis"].value_counts(dropna=False).to_dict() if not out.empty else {},
        "lc_exists_counts": out["lc_exists"].value_counts(dropna=False).astype(int).to_dict() if not out.empty else {},
        "lc_coord_status_counts": out["lc_coord_status"].value_counts(dropna=False).to_dict() if not out.empty else {},
        "masked_index_status_counts": out["masked_index_status"].value_counts(dropna=False).to_dict() if not out.empty else {},
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Audited missing-coordinate rows: {len(missing)}")
    print(f"Wrote row audit: {args.output}")
    print(f"Wrote summary: {args.summary}")
    print("Diagnosis counts:")
    for key, value in summary["diagnosis_counts"].items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
