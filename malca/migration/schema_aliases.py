from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import re
import shutil
from typing import Literal

import numpy as np
import pandas as pd

from malca.candidates import ensure_candidate_id
from malca.product_schema import TIMESCALE_LTV, TIMESCALE_STV
from malca.table_io import read_parquet_table, write_parquet_table


PreferPolicy = Literal["fail", "canonical", "legacy"]


COMMON_ALIAS_MAP: dict[str, str] = {
    "ra_deg": "ra",
    "dec_deg": "dec",
    "gaia_source_id": "source_id",
    "gaia_phot_g_mean_mag": "phot_g_mean_mag",
    "gaia_bp_mag": "phot_bp_mean_mag",
    "gaia_rp_mag": "phot_rp_mean_mag",
    "gaia_parallax": "parallax",
    "gaia_pmra": "pmra",
    "gaia_pmdec": "pmdec",
    "gaia_pm_total": "pm_total",
    "M_G": "mg",
    "M_G0": "mg0",
    "bp_rp0": "bprp0",
    "ltv_filter_reason": "filter_reason",
}

STV_ALIAS_MAP: dict[str, str] = {
    "path": "lc_path",
    "dat_path": "lc_path",
}

LTV_CORE_ALIAS_MAP: dict[str, str] = {
    "Pstarss gmag": "baseline_mag",
    "Median": "ltv_median",
    "Median_err": "ltv_median_err",
    "Dispersion": "ltv_dispersion",
    "Slope": "ltv_slope",
    "Quad Slope": "ltv_slope_quad",
    "max diff": "ltv_max_diff",
    "n_seasons": "ltv_n_seasons",
    "ls_period": "ltv_ls_period",
    "ls_power": "ltv_ls_power",
    "ls_fap": "ltv_ls_fap",
    "coeff1": "ltv_coeff1",
    "coeff2": "ltv_coeff2",
    "vg_has_v": "ltv_vg_has_v",
    "vg_overlap_days": "ltv_vg_overlap_days",
    "vg_overlap_fraction": "ltv_vg_overlap_fraction",
    "season_points_min": "ltv_season_points_min",
    "season_points_median": "ltv_season_points_median",
    "season_points_max": "ltv_season_points_max",
    "season_span_days_mean": "ltv_season_span_days_mean",
    "season_span_days_median": "ltv_season_span_days_median",
    "season_span_days_max": "ltv_season_span_days_max",
    "season_step_max_mag": "ltv_season_step_max_mag",
    "season_step_mean_abs_mag": "ltv_season_step_mean_abs_mag",
    "season_step_max_fraction": "ltv_season_step_max_fraction",
    "season_monotonicity_fraction": "ltv_season_monotonicity_fraction",
    "season_spearman_rho": "ltv_season_spearman_rho",
    "season_kendall_tau": "ltv_season_kendall_tau",
    "leave1out_slope_std": "ltv_leave1out_slope_std",
    "leave1out_slope_range": "ltv_leave1out_slope_range",
    "trend_slope_mag_per_year": "ltv_trend_slope_mag_per_year",
    "trend_quad_mag_per_year2": "ltv_trend_quad_mag_per_year2",
    "trend_slope_err_mag_per_year": "ltv_trend_slope_err_mag_per_year",
    "trend_slope_snr": "ltv_trend_slope_snr",
    "trend_r2": "ltv_trend_r2",
    "trend_delta_bic_linear": "ltv_trend_delta_bic_linear",
    "trend_delta_bic_quadratic": "ltv_trend_delta_bic_quadratic",
}

LTV_PIPELINE_ALIAS_MAP: dict[str, str] = {
    "neowise_n_epochs": "ltv_neowise_n_epochs",
    "w1_slope": "ltv_neowise_w1_slope",
    "w1_w2_slope": "ltv_neowise_w1_w2_slope",
    "dust_candidate": "ltv_dust_candidate",
    "dust_excess": "ltv_dust_excess",
    "dust_trend_class": "ltv_dust_trend_class",
    "dust_trend_flag": "ltv_dust_trend_flag",
    "stoch_sf_ml_amplitude": "ltv_stoch_sf_ml_amplitude",
    "stoch_sf_ml_gamma": "ltv_stoch_sf_ml_gamma",
    "stoch_iar_phi": "ltv_stoch_iar_phi",
    "stoch_mhps_high": "ltv_stoch_mhps_high",
    "stoch_mhps_low": "ltv_stoch_mhps_low",
    "stoch_mhps_non_zero": "ltv_stoch_mhps_non_zero",
    "stoch_mhps_pn_flag": "ltv_stoch_mhps_pn_flag",
    "stoch_mhps_ratio": "ltv_stoch_mhps_ratio",
    "stoch_gp_drw_sigma": "ltv_stoch_gp_drw_sigma",
    "stoch_gp_drw_tau": "ltv_stoch_gp_drw_tau",
}

LTV_DROP_COLUMNS: tuple[str, ...] = ("ltv_passed_filters",)

CAMERA_FIELD_ALIAS_MAP: dict[str, str] = {
    "camera_field_key": "camera_name_key",
    "camera_fields": "camera_names",
    "camera_field_count": "camera_name_count",
    "camera_field_key_fraction": "camera_name_key_fraction",
    "stats_camera_field_key": "stats_camera_name_key",
    "stats_camera_fields": "stats_camera_names",
    "stats_camera_field_count": "stats_camera_name_count",
    "stats_camera_field_key_fraction": "stats_camera_name_key_fraction",
}

_LAYER_COLUMNS = ("lc_stats", "external_stats", "derived_stats")


@dataclass(frozen=True)
class SchemaScanResult:
    path: str
    kind: str
    timescale: str | None
    rows: int | None
    columns: list[str]
    legacy_columns: list[str]
    canonical_columns: list[str]
    needs_conversion: bool
    error: str | None = None


@dataclass(frozen=True)
class SchemaMigrationResult:
    input_path: str
    output_path: str | None
    kind: str
    timescale: str | None
    rows: int | None
    changed_columns: dict[str, str] = field(default_factory=dict)
    dropped_columns: list[str] = field(default_factory=list)
    added_columns: list[str] = field(default_factory=list)
    wrote: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _is_parquet_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".parquet"


def _is_parquet_dataset_dir(path: Path) -> bool:
    return path.is_dir() and any(child.is_file() and child.suffix.lower() == ".parquet" for child in path.glob("chunk_*.parquet"))


def _read_product(path: Path) -> tuple[pd.DataFrame, str]:
    if _is_parquet_file(path):
        return read_parquet_table(path), "file"
    if _is_parquet_dataset_dir(path):
        table = pd.read_parquet(path)
        return table.to_frame() if isinstance(table, pd.Series) else table, "dataset"
    raise ValueError(f"Not a MALCA parquet product file or chunked dataset: {path}")


def _write_product(df: pd.DataFrame, path: Path, *, kind: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists, use overwrite=True: {path}")
    if kind == "file":
        write_parquet_table(df, path)
        return
    if kind == "dataset":
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        path.mkdir(parents=True, exist_ok=True)
        write_parquet_table(df, path / "chunk_000000.parquet")
        return
    raise ValueError(f"Unknown product kind: {kind}")


def _legacy_map_for_timescale(timescale: str) -> dict[str, str]:
    mapping = dict(COMMON_ALIAS_MAP)
    if timescale == TIMESCALE_STV:
        mapping.update(STV_ALIAS_MAP)
    elif timescale == TIMESCALE_LTV:
        mapping.update(LTV_CORE_ALIAS_MAP)
        mapping.update(LTV_PIPELINE_ALIAS_MAP)
    else:
        raise ValueError(f"Unsupported timescale: {timescale}")
    return mapping


def detect_product_timescale(path: str | Path, df: pd.DataFrame, explicit: str | None = None) -> str:
    if explicit is not None:
        value = str(explicit).strip().lower()
        if value not in {TIMESCALE_STV, TIMESCALE_LTV}:
            raise ValueError(f"Unsupported timescale: {explicit}")
        return value

    columns = set(df.columns)
    if "timescale" in columns:
        values = df["timescale"].dropna().astype(str).str.strip().str.lower().unique()
        values = [value for value in values if value]
        if len(values) == 1 and values[0] in {TIMESCALE_STV, TIMESCALE_LTV}:
            return values[0]

    path_parts = {part.lower() for part in Path(path).parts}
    stv_score = 1 if TIMESCALE_STV in path_parts else 0
    ltv_score = 1 if TIMESCALE_LTV in path_parts else 0

    if columns & {"dip_significant", "jump_significant", "dip_best_t0", "jump_best_t0"}:
        stv_score += 4
    if columns & (set(LTV_CORE_ALIAS_MAP) | {"ltv_slope", "ltv_max_diff", "ltv_median"}):
        ltv_score += 4

    if stv_score > ltv_score:
        return TIMESCALE_STV
    if ltv_score > stv_score:
        return TIMESCALE_LTV
    raise ValueError(f"Could not infer product timescale for {path}; pass --timescale")


def _values_equal(left: pd.Series, right: pd.Series) -> bool:
    left_obj = left.astype("object")
    right_obj = right.astype("object")
    both_missing = left_obj.isna() & right_obj.isna()
    comparable = ~(both_missing)
    if not bool(comparable.any()):
        return True
    left_vals = left_obj[comparable].map(lambda value: "" if pd.isna(value) else str(value))
    right_vals = right_obj[comparable].map(lambda value: "" if pd.isna(value) else str(value))
    return bool(left_vals.equals(right_vals))


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    if value is pd.NA:
        return True
    if isinstance(value, float):
        return bool(np.isnan(value))
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    if missing is pd.NA:
        return True
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _camera_field_tokens(value: object) -> list[str]:
    if _is_missing_scalar(value):
        return []
    if isinstance(value, (list, tuple, set)):
        raw_tokens = list(value)
    else:
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            raw_tokens = parsed
        else:
            raw_tokens = re.split(r"[,;]", text)
    tokens: list[str] = []
    for token in raw_tokens:
        token_text = str(token).strip()
        if token_text:
            tokens.append(token_text)
    return tokens


def _camera_name_from_combined_token(token: object) -> str:
    text = "" if _is_missing_scalar(token) else str(token).strip()
    if "/" not in text:
        return ""
    return text.split("/", 1)[0].strip()


def _camera_name_key_value(value: object) -> object:
    if _is_missing_scalar(value):
        return value
    return _camera_name_from_combined_token(value)


def _camera_names_value(value: object) -> str:
    cameras = {
        camera
        for token in _camera_field_tokens(value)
        for camera in [_camera_name_from_combined_token(token)]
        if camera
    }
    return ",".join(sorted(cameras))


def _count_from_label_string(value: object) -> int | object:
    if _is_missing_scalar(value):
        return value
    labels = [token for token in _camera_field_tokens(value) if token]
    return len(labels)


def _camera_field_column_values(out: pd.DataFrame, src: str, dst: str) -> pd.Series:
    values = out[src]
    if src.endswith("_key"):
        return values.map(_camera_name_key_value)
    if src.endswith("_fields"):
        return values.map(_camera_names_value)
    if src.endswith("_count"):
        names_col = "stats_camera_names" if src.startswith("stats_") else "camera_names"
        if names_col in out.columns:
            return out[names_col].map(_count_from_label_string)
        return values
    if src.endswith("_fraction"):
        return values
    return values


def _assign_migrated_column(
    out: pd.DataFrame,
    *,
    src: str,
    dst: str,
    values: pd.Series,
    prefer: PreferPolicy,
) -> pd.DataFrame:
    if dst in out.columns:
        if not _values_equal(values, out[dst]):
            if prefer == "fail":
                raise ValueError(f"Conflicting legacy/canonical columns: {src} -> {dst}")
            if prefer == "legacy":
                out[dst] = values
        return out.drop(columns=[src])
    out[dst] = values
    return out.drop(columns=[src])


def migrate_camera_field_frame(
    df: pd.DataFrame,
    *,
    prefer: PreferPolicy = "fail",
) -> pd.DataFrame:
    """Rewrite legacy combined camera-field columns to split camera-name columns.

    This is a schema migration for saved products.  It can split values such as
    ``ba/F1`` into ``ba``.  If a legacy artifact only saved field-only labels
    such as ``F1``, the camera name is unrecoverable from that artifact and the
    migrated camera-name value is left blank.
    """
    out = df.copy()
    for src, dst in CAMERA_FIELD_ALIAS_MAP.items():
        if src not in out.columns:
            continue
        values = _camera_field_column_values(out, src, dst)
        out = _assign_migrated_column(out, src=src, dst=dst, values=values, prefer=prefer)

    for layer in _LAYER_COLUMNS:
        if layer not in out.columns:
            continue
        out[layer] = out[layer].map(lambda value: _json_dumps_mapping(migrate_camera_field_mapping(_parse_mapping(value))))
    return out


def _parse_mapping(value: object) -> dict[str, object]:
    if _is_missing_scalar(value):
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _json_dumps_mapping(value: dict[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _mapping_value_for_legacy_camera_field(mapping: dict[str, object], src: str) -> object:
    value = mapping.get(src)
    if src.endswith("_key"):
        return _camera_name_key_value(value)
    if src.endswith("_fields"):
        return _camera_names_value(value)
    if src.endswith("_count"):
        names_key = "stats_camera_names" if src.startswith("stats_") else "camera_names"
        if names_key in mapping:
            return _count_from_label_string(mapping.get(names_key))
        return value
    return value


def migrate_camera_field_mapping(
    mapping: dict[str, object],
    *,
    prefer: PreferPolicy = "legacy",
) -> dict[str, object]:
    """Rewrite legacy camera-field keys in a row/payload mapping."""
    out = dict(mapping)
    for layer in _LAYER_COLUMNS:
        layer_payload = _parse_mapping(out.get(layer))
        if layer_payload:
            out[layer] = migrate_camera_field_mapping(layer_payload, prefer=prefer)

    for src, dst in CAMERA_FIELD_ALIAS_MAP.items():
        if src not in out:
            continue
        value = _mapping_value_for_legacy_camera_field(out, src)
        if dst not in out or prefer == "legacy":
            out[dst] = value
        out.pop(src, None)
    return out


def _apply_alias_map(
    df: pd.DataFrame,
    mapping: dict[str, str],
    *,
    prefer: PreferPolicy,
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    out = df.copy()
    changed: dict[str, str] = {}
    dropped: list[str] = []
    for src, dst in mapping.items():
        if src not in out.columns:
            continue
        if dst in out.columns:
            if _values_equal(out[src], out[dst]):
                out = out.drop(columns=[src])
                dropped.append(src)
                changed[src] = dst
                continue
            if prefer == "fail":
                raise ValueError(f"Conflicting legacy/canonical columns: {src} -> {dst}")
            if prefer == "legacy":
                out[dst] = out[src]
            out = out.drop(columns=[src])
            dropped.append(src)
            changed[src] = dst
            continue
        out = out.rename(columns={src: dst})
        changed[src] = dst
    return out, changed, dropped


def _derive_common_columns(df: pd.DataFrame, timescale: str) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    added: list[str] = []

    if "source_id" in out.columns and "gaia_id" not in out.columns:
        out["gaia_id"] = out["source_id"]
        added.append("gaia_id")

    if "bp_rp" not in out.columns and {"phot_bp_mean_mag", "phot_rp_mean_mag"}.issubset(out.columns):
        bp = pd.to_numeric(out["phot_bp_mean_mag"], errors="coerce")
        rp = pd.to_numeric(out["phot_rp_mean_mag"], errors="coerce")
        out["bp_rp"] = bp - rp
        added.append("bp_rp")

    if "pm_total" not in out.columns and {"pmra", "pmdec"}.issubset(out.columns):
        pmra = pd.to_numeric(out["pmra"], errors="coerce")
        pmdec = pd.to_numeric(out["pmdec"], errors="coerce")
        out["pm_total"] = np.sqrt(pmra * pmra + pmdec * pmdec)
        added.append("pm_total")

    if "time_span_days" not in out.columns and {"jd_first", "jd_last"}.issubset(out.columns):
        jd_first = pd.to_numeric(out["jd_first"], errors="coerce")
        jd_last = pd.to_numeric(out["jd_last"], errors="coerce")
        out["time_span_days"] = jd_last - jd_first
        added.append("time_span_days")

    if "failed_any" not in out.columns:
        out["failed_any"] = False
        added.append("failed_any")
    if "filter_reason" not in out.columns:
        out["filter_reason"] = pd.NA
        added.append("filter_reason")

    out["timescale"] = timescale
    if "timescale" not in added:
        added.append("timescale")
    return out, added


def convert_product_frame(
    df: pd.DataFrame,
    timescale: str,
    product_kind: str | None = None,
    *,
    prefer: PreferPolicy = "fail",
) -> pd.DataFrame:
    del product_kind
    timescale = str(timescale).strip().lower()
    if timescale not in {TIMESCALE_STV, TIMESCALE_LTV}:
        raise ValueError(f"Unsupported timescale: {timescale}")

    out, _changed, _dropped = _apply_alias_map(df, _legacy_map_for_timescale(timescale), prefer=prefer)
    out = migrate_camera_field_frame(out, prefer=prefer)

    if timescale == TIMESCALE_LTV:
        drop_cols = [col for col in LTV_DROP_COLUMNS if col in out.columns]
        if drop_cols:
            out = out.drop(columns=drop_cols)

    out, _added = _derive_common_columns(out, timescale)
    out = ensure_candidate_id(
        out,
        prefix=timescale,
        source_cols=("candidate_id", "asas_sn_id", "source_id", "lc_path"),
    )
    return out


def _conversion_details(
    before: pd.DataFrame,
    after: pd.DataFrame,
    timescale: str,
) -> tuple[dict[str, str], list[str], list[str]]:
    mapping = _legacy_map_for_timescale(timescale)
    mapping.update(CAMERA_FIELD_ALIAS_MAP)
    changed = {src: dst for src, dst in mapping.items() if src in before.columns and dst in after.columns}
    dropped = [col for col in before.columns if col not in after.columns]
    added = [col for col in after.columns if col not in before.columns]
    return changed, dropped, added


def scan_product(path: str | Path, *, timescale: str | None = None) -> SchemaScanResult:
    product_path = Path(path).expanduser()
    try:
        df, kind = _read_product(product_path)
        detected = detect_product_timescale(product_path, df, explicit=timescale)
        mapping = _legacy_map_for_timescale(detected)
        mapping.update(CAMERA_FIELD_ALIAS_MAP)
        legacy = [col for col in df.columns if col in mapping or col in LTV_DROP_COLUMNS]
        canonical = [col for col in df.columns if col in set(mapping.values()) or col in {"candidate_id", "timescale", "lc_path"}]
        return SchemaScanResult(
            path=str(product_path),
            kind=kind,
            timescale=detected,
            rows=len(df),
            columns=list(df.columns),
            legacy_columns=legacy,
            canonical_columns=canonical,
            needs_conversion=bool(legacy) or "candidate_id" not in df.columns or "timescale" not in df.columns,
        )
    except Exception as exc:
        return SchemaScanResult(
            path=str(product_path),
            kind="unknown",
            timescale=timescale,
            rows=None,
            columns=[],
            legacy_columns=[],
            canonical_columns=[],
            needs_conversion=False,
            error=str(exc),
        )


def convert_product_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    timescale: str | None = None,
    overwrite: bool = False,
    prefer: PreferPolicy = "fail",
) -> SchemaMigrationResult:
    in_path = Path(input_path).expanduser()
    out_path = Path(output_path).expanduser()
    try:
        df, kind = _read_product(in_path)
        detected = detect_product_timescale(in_path, df, explicit=timescale)
        converted = convert_product_frame(df, detected, kind, prefer=prefer)
        changed, dropped, added = _conversion_details(df, converted, detected)
        _write_product(converted, out_path, kind=kind, overwrite=overwrite)
        return SchemaMigrationResult(
            input_path=str(in_path),
            output_path=str(out_path),
            kind=kind,
            timescale=detected,
            rows=len(converted),
            changed_columns=changed,
            dropped_columns=dropped,
            added_columns=added,
            wrote=True,
        )
    except Exception as exc:
        return SchemaMigrationResult(
            input_path=str(in_path),
            output_path=str(out_path),
            kind="unknown",
            timescale=timescale,
            rows=None,
            error=str(exc),
        )


def _iter_product_paths(root: Path) -> list[Path]:
    root = root.expanduser()
    if _is_parquet_file(root) or _is_parquet_dataset_dir(root):
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Input not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Expected parquet file, chunked dataset, or directory: {root}")

    products: list[Path] = []
    for path in sorted(root.rglob("*")):
        if any(parent in products for parent in path.parents):
            continue
        if _is_parquet_dataset_dir(path):
            products.append(path)
        elif _is_parquet_file(path):
            products.append(path)
    return products


def _relative_output_path(input_path: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file() or _is_parquet_dataset_dir(input_root):
        return output_root / input_path.name
    return output_root / input_path.relative_to(input_root)


def convert_run_tree(
    input_root: str | Path,
    output_root: str | Path,
    *,
    timescale: str | None = None,
    overwrite: bool = False,
    prefer: PreferPolicy = "fail",
) -> list[SchemaMigrationResult]:
    in_root = Path(input_root).expanduser()
    out_root = Path(output_root).expanduser()
    products = _iter_product_paths(in_root)
    results: list[SchemaMigrationResult] = []
    for product in products:
        out_path = _relative_output_path(product, in_root, out_root)
        results.append(
            convert_product_file(
                product,
                out_path,
                timescale=timescale,
                overwrite=overwrite,
                prefer=prefer,
            )
        )
    return results


def write_migration_report(results: list[SchemaMigrationResult | SchemaScanResult], path: str | Path) -> None:
    report_path = Path(path).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(item) if not isinstance(item, SchemaMigrationResult) else item.to_dict() for item in results]
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="ascii")
