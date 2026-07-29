from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from malca.products.candidates import ensure_candidate_id, validate_candidate_ids
from malca.products.feature_layers import (
    ALL_FEATURE_LAYER_COLUMNS,
    FEATURE_LAYER_COLUMNS,
    feature_layer_for_column,
    non_layer_feature_columns,
    parse_layer_value,
    split_layer_path,
)


TIMESCALE_STV = "stv"
TIMESCALE_LTV = "ltv"

IDENTITY_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "timescale",
    "asas_sn_id",
    "lc_path",
)

SKY_GAIA_COLUMNS: tuple[str, ...] = (
    "gaia_id",
    "source_id",
    "gaia_dr2_id",
    "gaia_id_release",
    "gaia_id_mapping_status",
    "dr2_dr3_angular_distance_mas",
    "dr2_dr3_magnitude_difference",
    "ra",
    "dec",
)

LIGHTCURVE_BASIC_COLUMNS: tuple[str, ...] = (
    "n_points",
    "jd_first",
    "jd_last",
    "time_span_days",
    "n_unique_nights",
    "baseline_mag",
    "baseline_source",
)

FILTER_COLUMNS: tuple[str, ...] = (
    "failed_any",
    "filter_reason",
)

GAIA_CONTEXT_COLUMNS: tuple[str, ...] = (
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "bp_rp",
    "distance_gspphot",
    "parallax",
    "parallax_error",
    "pmra",
    "pmdec",
    "pm_total",
    "ruwe",
    "high_pm_flag",
    "high_ruwe_flag",
)

CMD_COLUMNS: tuple[str, ...] = (
    "A_v_3d",
    "ebv_3d",
    "mg",
    "mg0",
    "bprp0",
    "derived_bp_rp",
    "derived_j_k",
    "derived_mrp",
    "derived_mks",
    "derived_wrp",
    "derived_wjk",
)

SHARED_PRODUCT_COLUMNS: tuple[str, ...] = (
    *IDENTITY_COLUMNS,
    *SKY_GAIA_COLUMNS,
    *LIGHTCURVE_BASIC_COLUMNS,
    *FILTER_COLUMNS,
    *GAIA_CONTEXT_COLUMNS,
    *ALL_FEATURE_LAYER_COLUMNS,
    *CMD_COLUMNS,
)

STV_REQUIRED_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "timescale",
    "lc_path",
)

STV_EVENT_REQUIRED_COLUMNS: tuple[str, ...] = (
    *STV_REQUIRED_COLUMNS,
    "event_schema_version",
    "event_score_version",
    "tag_stats_status",
    "tag_stats_error",
    "tag_stats_version",
    "raw_n_points",
    "clean_n_points",
    "raw_n_cameras",
    "raw_camera_ids",
    "raw_asassn_fields",
    "raw_camera_names",
    "baseline_cross_band_calibrated",
    "baseline_cross_band_details",
    "dip_best_delta_mag",
    "jump_best_delta_mag",
)

LTV_REQUIRED_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "timescale",
    "asas_sn_id",
    "lc_path",
    "ra",
    "dec",
)

STV_FORBIDDEN_PRODUCT_COLUMNS: tuple[str, ...] = ("path", "dat_path", "ra_deg", "dec_deg")
LTV_FORBIDDEN_PRODUCT_COLUMNS: tuple[str, ...] = (
    "ra_deg",
    "dec_deg",
    "Pstarss gmag",
    "Median",
    "Median_err",
    "Dispersion",
    "Slope",
    "Quad Slope",
    "max diff",
    "n_seasons",
    "ls_period",
    "ls_power",
    "ls_fap",
    "coeff1",
    "coeff2",
    "vg_has_v",
    "vg_overlap_days",
    "vg_overlap_fraction",
    "season_points_min",
    "season_points_median",
    "season_points_max",
    "season_span_days_mean",
    "season_span_days_median",
    "season_span_days_max",
    "season_step_max_mag",
    "season_step_mean_abs_mag",
    "season_step_max_fraction",
    "season_monotonicity_fraction",
    "season_spearman_rho",
    "season_kendall_tau",
    "leave1out_slope_std",
    "leave1out_slope_range",
    "trend_slope_mag_per_year",
    "trend_quad_mag_per_year2",
    "trend_slope_err_mag_per_year",
    "trend_slope_snr",
    "trend_r2",
    "trend_delta_bic_linear",
    "trend_delta_bic_quadratic",
    "w1_slope",
    "w1_w2_slope",
    "neowise_n_epochs",
    "dust_candidate",
    "dust_excess",
    "dust_trend_class",
    "dust_trend_flag",
    "stoch_sf_ml_amplitude",
    "stoch_sf_ml_gamma",
    "stoch_iar_phi",
    "stoch_mhps_high",
    "stoch_mhps_low",
    "stoch_mhps_non_zero",
    "stoch_mhps_pn_flag",
    "stoch_mhps_ratio",
    "stoch_gp_drw_sigma",
    "stoch_gp_drw_tau",
    "ltv_filter_reason",
    "ltv_passed_filters",
    "M_G",
    "M_G0",
    "bp_rp0",
    "gaia_source_id",
    "gaia_pmra",
    "gaia_pmdec",
    "gaia_pm_total",
    "gaia_phot_g_mean_mag",
    "gaia_bp_mag",
    "gaia_rp_mag",
)


@dataclass(frozen=True)
class ProductSchemaError(ValueError):
    timescale: str
    stage: str | None
    missing: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    message: str = ""

    def __str__(self) -> str:
        parts = [self.message or f"{self.timescale.upper()} product schema check failed"]
        if self.stage:
            parts.append(f"stage={self.stage}")
        if self.missing:
            parts.append("missing=" + ",".join(self.missing))
        if self.forbidden:
            parts.append("forbidden=" + ",".join(self.forbidden))
        return "; ".join(parts)


def _missing_columns(df: pd.DataFrame, required: Iterable[str]) -> tuple[str, ...]:
    return tuple(column for column in required if not _has_required_column_or_feature(df, str(column)))


def _has_required_column_or_feature(df: pd.DataFrame, column: str) -> bool:
    if column in df.columns:
        return True
    if "." in column:
        try:
            layer, key = split_layer_path(column)
        except ValueError:
            return False
    else:
        layer = feature_layer_for_column(column)
        if layer is None:
            return False
        key = column
    if layer not in df.columns:
        return False
    if df.empty:
        return True
    # A layer key is part of the row schema, not a table-level suggestion.  An
    # ``any`` check allowed one populated row to make every other row appear
    # schema-complete.
    return bool(df[layer].map(lambda value: key in parse_layer_value(value)).all())


def _present_columns(df: pd.DataFrame, forbidden: Iterable[str]) -> tuple[str, ...]:
    return tuple(column for column in forbidden if column in df.columns)


def _ltv_forbidden_columns() -> tuple[str, ...]:
    from malca.enrichment.external_lcs import EXTERNAL_LC_COLUMNS

    allowed_external_lc_columns = set(EXTERNAL_LC_COLUMNS)
    return tuple(
        column
        for column in LTV_FORBIDDEN_PRODUCT_COLUMNS
        if column not in allowed_external_lc_columns
    )


def _assert_timescale_values(df: pd.DataFrame, timescale: str, *, stage: str | None) -> None:
    if "timescale" not in df.columns:
        return
    values = df["timescale"].astype("string").str.strip().str.lower()
    bad = values[values.isna() | values.eq("") | values.ne(timescale)]
    if not bad.empty:
        raise ProductSchemaError(
            timescale=timescale,
            stage=stage,
            message=f"Unexpected timescale values for {timescale.upper()} product",
        )


def _assert_required_identity_values(
    df: pd.DataFrame,
    required: Iterable[str],
    *,
    timescale: str,
    stage: str | None,
) -> None:
    required_set = {str(column) for column in required}
    if "candidate_id" in required_set and "candidate_id" in df.columns:
        try:
            validate_candidate_ids(df, key_col="candidate_id", require_unique=True)
        except ValueError as exc:
            raise ProductSchemaError(
                timescale=timescale,
                stage=stage,
                message=str(exc),
            ) from exc

    conditional_tag_error = {"tag_stats_status", "tag_stats_error"}.issubset(required_set)
    for column in sorted(required_set - {"candidate_id"}):
        if df.empty or (conditional_tag_error and column == "tag_stats_error"):
            continue
        values = _required_value_series(df, column)
        if values is None:
            # Missing keys/columns are reported by ``_missing_columns`` before
            # value validation.  Keep this helper focused on present values.
            continue
        missing = values.isna()
        if pd.api.types.is_object_dtype(values) or pd.api.types.is_string_dtype(values):
            missing = missing | values.astype("string").str.strip().eq("").fillna(True)
        if bool(missing.any()):
            raise ProductSchemaError(
                timescale=timescale,
                stage=stage,
                message=f"Required field {column!r} contains {int(missing.sum())} missing value(s)",
            )

    if conditional_tag_error and not df.empty:
        _assert_tag_stats_error_values(
            df,
            timescale=timescale,
            stage=stage,
        )


def _required_value_series(df: pd.DataFrame, column: str) -> pd.Series | None:
    """Resolve a required value identically from a flat or layer-first frame."""
    if column in df.columns:
        return df[column]
    if "." in column:
        try:
            layer, key = split_layer_path(column)
        except ValueError:
            return None
    else:
        layer = feature_layer_for_column(column)
        if layer is None:
            return None
        key = column
    if layer not in df.columns:
        return None
    return df[layer].map(lambda value: parse_layer_value(value).get(key, pd.NA))


def _assert_tag_stats_error_values(
    df: pd.DataFrame,
    *,
    timescale: str,
    stage: str | None,
) -> None:
    """Validate the status-dependent meaning of ``tag_stats_error``.

    Successful rows must carry the key with an explicitly blank error.  Any
    other terminal status must carry a non-blank diagnostic.  This contract is
    deliberately evaluated through ``_required_value_series`` so flat and
    layer-first products behave the same way.
    """
    statuses = _required_value_series(df, "tag_stats_status")
    errors = _required_value_series(df, "tag_stats_error")
    if statuses is None or errors is None:
        return

    status_text = statuses.astype("string").str.strip().str.lower()
    error_text = errors.astype("string").str.strip()
    error_missing = errors.isna()
    status_ok = status_text.eq("ok").fillna(False)
    error_blank = error_text.eq("").fillna(False)

    invalid_ok = status_ok & (error_missing | ~error_blank)
    invalid_non_ok = ~status_ok & (error_missing | error_blank)
    invalid = invalid_ok | invalid_non_ok
    if bool(invalid.any()):
        raise ProductSchemaError(
            timescale=timescale,
            stage=stage,
            message=(
                "tag_stats_error must be explicitly blank when tag_stats_status='ok' "
                "and non-blank otherwise; "
                f"found {int(invalid.sum())} invalid row(s)"
            ),
        )


def _assert_layer_values(df: pd.DataFrame, *, timescale: str, stage: str | None) -> None:
    for layer in FEATURE_LAYER_COLUMNS:
        if layer not in df.columns:
            continue
        invalid: list[object] = []
        for idx, value in df[layer].items():
            if isinstance(value, dict):
                continue
            if isinstance(value, str):
                text = value.strip()
                if text.startswith("{") and text.endswith("}"):
                    try:
                        import json

                        parsed = json.loads(text)
                    except Exception:
                        parsed = None
                    if isinstance(parsed, dict):
                        continue
            invalid.append(idx)
        if invalid:
            raise ProductSchemaError(
                timescale=timescale,
                stage=stage,
                message=f"Feature layer {layer!r} contains invalid JSON/mapping values at {invalid[:5]}",
            )


def assert_candidate_product_schema(
    df: pd.DataFrame,
    *,
    timescale: str,
    stage: str | None = None,
    required: Iterable[str] | None = None,
    forbidden: Iterable[str] = (),
) -> None:
    required_columns = tuple(required or ())
    missing = _missing_columns(df, (*FEATURE_LAYER_COLUMNS, *required_columns))
    present_forbidden = _present_columns(df, forbidden)
    flat_features = tuple(non_layer_feature_columns(df.columns))
    present_forbidden = tuple(dict.fromkeys((*present_forbidden, *flat_features)))
    if missing or present_forbidden:
        raise ProductSchemaError(
            timescale=timescale,
            stage=stage,
            missing=missing,
            forbidden=present_forbidden,
        )
    _assert_timescale_values(df, timescale, stage=stage)
    _assert_required_identity_values(
        df,
        required_columns,
        timescale=timescale,
        stage=stage,
    )
    _assert_layer_values(df, timescale=timescale, stage=stage)


def assert_stv_product_schema(
    df: pd.DataFrame,
    *,
    stage: str | None = None,
    required: Iterable[str] | None = None,
) -> None:
    assert_candidate_product_schema(
        df,
        timescale=TIMESCALE_STV,
        stage=stage,
        required=required or STV_REQUIRED_COLUMNS,
        forbidden=STV_FORBIDDEN_PRODUCT_COLUMNS,
    )


def assert_ltv_product_schema(
    df: pd.DataFrame,
    *,
    stage: str | None = None,
    required: Iterable[str] | None = None,
) -> None:
    assert_candidate_product_schema(
        df,
        timescale=TIMESCALE_LTV,
        stage=stage,
        required=required or LTV_REQUIRED_COLUMNS,
        forbidden=_ltv_forbidden_columns(),
    )


def add_canonical_identity(
    df: pd.DataFrame,
    *,
    timescale: str,
    source_cols: tuple[str, ...] = ("candidate_id", "asas_sn_id", "source_id", "lc_path"),
) -> pd.DataFrame:
    """Return a copy with canonical timescale and candidate_id columns."""
    out = ensure_candidate_id(df, prefix=timescale, source_cols=source_cols)
    out["timescale"] = timescale
    return out


def add_stv_identity(df: pd.DataFrame) -> pd.DataFrame:
    return add_canonical_identity(
        df,
        timescale=TIMESCALE_STV,
        source_cols=("candidate_id", "asas_sn_id", "source_id", "lc_path"),
    )


def add_ltv_identity(df: pd.DataFrame) -> pd.DataFrame:
    return add_canonical_identity(
        df,
        timescale=TIMESCALE_LTV,
        source_cols=("candidate_id", "asas_sn_id", "source_id", "lc_path"),
    )
