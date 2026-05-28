from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from malca.candidates import ensure_candidate_id


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
)

SHARED_PRODUCT_COLUMNS: tuple[str, ...] = (
    *IDENTITY_COLUMNS,
    *SKY_GAIA_COLUMNS,
    *LIGHTCURVE_BASIC_COLUMNS,
    *FILTER_COLUMNS,
    *GAIA_CONTEXT_COLUMNS,
    *CMD_COLUMNS,
)

STV_REQUIRED_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "timescale",
    "lc_path",
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
    return tuple(column for column in required if column not in df.columns)


def _present_columns(df: pd.DataFrame, forbidden: Iterable[str]) -> tuple[str, ...]:
    return tuple(column for column in forbidden if column in df.columns)


def _assert_timescale_values(df: pd.DataFrame, timescale: str, *, stage: str | None) -> None:
    if "timescale" not in df.columns:
        return
    values = df["timescale"].dropna().astype(str).str.strip().str.lower()
    bad = values[values != timescale]
    if not bad.empty:
        raise ProductSchemaError(
            timescale=timescale,
            stage=stage,
            message=f"Unexpected timescale values for {timescale.upper()} product",
        )


def assert_candidate_product_schema(
    df: pd.DataFrame,
    *,
    timescale: str,
    stage: str | None = None,
    required: Iterable[str] | None = None,
    forbidden: Iterable[str] = (),
) -> None:
    missing = _missing_columns(df, required or ())
    present_forbidden = _present_columns(df, forbidden)
    if missing or present_forbidden:
        raise ProductSchemaError(
            timescale=timescale,
            stage=stage,
            missing=missing,
            forbidden=present_forbidden,
        )
    _assert_timescale_values(df, timescale, stage=stage)


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
        forbidden=LTV_FORBIDDEN_PRODUCT_COLUMNS,
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
