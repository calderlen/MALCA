# This file was mechanically split from malca.review.app; preserve behavior when editing.
@lru_cache(maxsize=8)
def _load_run_params_meta_for_plot_dir(plot_dir: str | None) -> tuple[dict | None, str, str]:
    """Load run_params with status/meta from active plot directory."""
    if not plot_dir:
        return None, "missing", "No plot directory set"
    run_params_path = Path(plot_dir).resolve().parent / "run_params.json"
    if not run_params_path.exists():
        return None, "missing", f"Missing {run_params_path}"
    try:
        with open(run_params_path) as f:
            data = json.load(f)
    except Exception as exc:
        return None, "invalid", f"Could not parse run_params.json: {exc}"
    if not isinstance(data, dict):
        return None, "invalid", "run_params.json is not a JSON object"
    return data, "loaded", f"Loaded from {run_params_path}"


def _load_run_params_for_plot_dir(plot_dir: str | None) -> dict:
    """Load run_params.json near the active plot directory."""
    run_params, _status, _msg = _load_run_params_meta_for_plot_dir(plot_dir)
    return run_params or {}


def _lazy_panel_placeholder(message: str, tone: str = "muted") -> html.Div:
    """Small visible placeholder for lazy-loaded collapsible panels."""
    cls = "lazy-panel-placeholder"
    if tone:
        cls += f" lazy-panel-placeholder-{tone}"
    return html.Div(str(message), className=cls)


def _copyable_math_value(value: object) -> html.Div:
    """Render a MathJax value with a raw-value clipboard affordance."""
    raw = str(value)
    return html.Div([
        dcc.Markdown(raw, className='meta-field-value', mathjax=True),
        html.Button(
            "⧉",
            type="button",
            title="Copy raw value",
            className="metadata-copy-btn",
            **{
                "aria-label": "Copy raw value",
                "data-copy-text": raw,
            },
        ),
    ], className="copyable-math-field")


_REVIEW_SECTION_ORDER = [
    "Review Summary",
    "Dip Evidence",
    "Jump Evidence",
    "Coverage & Photometry",
    "Catalog & Vetting",
    "Classification & Environment",
    "Advanced Metadata",
]


def _truthy_display(value: object) -> bool | None:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _numeric_payload_value(payload: dict, key: str) -> float:
    try:
        value = payload.get(key)
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _copyable_plain_value(value: object) -> html.Div:
    raw = str(value)
    return html.Div([
        html.Span(raw, className="meta-field-value"),
        html.Button(
            "⧉",
            type="button",
            title="Copy raw value",
            className="metadata-copy-btn",
            **{
                "aria-label": "Copy raw value",
                "data-copy-text": raw,
            },
        ),
    ], className="copyable-math-field")


def _render_feature_section(title: str, rows: list[dict[str, str]], open_default: bool = False) -> html.Details | None:
    """Render a review-oriented metadata section."""
    if not rows:
        return None
    field_divs = [
        html.Div([
            html.Span(str(row.get("label", "")), className="meta-field-label"),
            _copyable_plain_value(row.get("value", "")),
        ], className="meta-field-row", title=str(row.get("key", "")))
        for row in rows
    ]
    slug = title.lower().replace("&", "and").replace(" ", "-")
    return html.Details([
        html.Summary(f"{title} ({len(rows)})"),
        html.Div(field_divs, className="meta-grid review-feature-grid"),
    ],
        id={"type": "meta-details", "group": title},
        open=bool(open_default),
        className=f"review-feature-section review-feature-section-{slug}",
    )


def _render_dipper_probability_card(payload: dict) -> html.Div | None:
    raw = payload.get("prob_dipper_like")
    if raw is None:
        return None
    try:
        prob = float(raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(prob):
        return None
    prob = min(max(float(prob), 0.0), 1.0)
    pct = 100.0 * prob
    return html.Div([
        html.Div("ML Dipper Probability", className="dipper-prob-card-label"),
        html.Div(f"{pct:.1f}%", className="dipper-prob-card-value"),
        html.Div(f"prob_dipper_like = {prob:.6f}", className="dipper-prob-card-detail"),
    ], className="dipper-prob-card")


def _stat_feature_label(key: str) -> str:
    raw = str(key)
    if raw.startswith("stats_"):
        raw = raw[6:]
    elif raw.startswith("ltv_"):
        raw = raw[4:]
    return raw.replace("_", " ").strip().title()


def _stat_feature_rows(stat_rows: list[tuple[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key_raw, value in stat_rows or []:
        key = str(key_raw)
        rows.append({
            "section": "Advanced Stats",
            "source_group": "Stats",
            "label": _stat_feature_label(key),
            "key": key,
            "value": str(value),
            "role": "stat",
        })
    return rows


_OTHER_PAYLOAD_CONTAINER_KEYS = {
    "payload",
    "payload_json",
    "lc_stats",
    "external_stats",
    "derived_stats",
}

_OTHER_PAYLOAD_RECORD_RUN_CONTEXT_KEYS = {
    "candidate_id",
    "feature_layer_version",
    "imported_at",
    "lc_path",
    "source_path",
    "trigger_mode",
}

_OTHER_PAYLOAD_COORDINATES_GAIA_KEYS = {
    "source_id",
    "ra_deg",
    "dec_deg",
    "ra_gaia",
    "dec_gaia",
    "parallax_gaia",
    "parallax_error",
    "parallax_error_gaia",
    "pmra_gaia",
    "pmdec_gaia",
    "ruwe_gaia",
    "radial_velocity_gaia",
    "rv_amplitude_robust_gaia",
    "ag_gspphot",
    "distance_gspphot_gaia",
    "teff_gspphot_gaia",
    "logg_gspphot_gaia",
    "mh_gspphot_gaia",
}

_OTHER_PAYLOAD_PERIOD_FILTER_KEYS = {
    "failed_gaia_ruwe",
    "failed_periodic_catalog",
    "period_conflict_flag",
    "period_consensus_support",
    "period_ogle_match",
    "period_primary_source",
    "period_source_periods",
    "periodic_flag",
}

_OTHER_PAYLOAD_EXTERNAL_LC_PREFIXES = (
    "asas3_lc_",
    "crts_lc_",
    "dasch_lc_",
    "gaia_epoch_lc_",
    "kelt_lc_",
    "nsvs_lc_",
    "ps1_lc_",
    "superwasp_lc_",
    "tess_",
    "ztf_lc_",
)

_OTHER_PAYLOAD_WISE_ERROR_KEYS = {"w1_err", "w2_err", "w3_err", "w4_err"}

_OTHER_PAYLOAD_SUBSECTION_ORDER = [
    "Record & Run Context",
    "Coordinates & Gaia",
    "Period & Filter Flags",
    "Derived Feature Extras",
    "Enrichment Stage Status",
    "SED Pipeline Status",
    "External LC Coverage",
    "Multi-Survey Features",
    "Photometric Error Columns",
    "Miscellaneous",
]

_OTHER_PAYLOAD_SUBSECTION_INDEX = {
    subsection: idx
    for idx, subsection in enumerate(_OTHER_PAYLOAD_SUBSECTION_ORDER)
}


def _payload_value_is_present(value: object) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _payload_field_label(key: str) -> str:
    token_map = {
        "id": "ID",
        "jd": "JD",
        "ra": "RA",
        "dec": "Dec",
        "lc": "LC",
        "lsp": "LSP",
        "pdm": "PDM",
        "ce": "CE",
        "yso": "YSO",
        "vsx": "VSX",
        "ztf": "ZTF",
        "gaia": "Gaia",
        "asas": "ASAS",
        "sn": "SN",
    }
    parts = []
    for token in str(key).replace("-", "_").split("_"):
        if not token:
            continue
        parts.append(token_map.get(token.lower(), token.capitalize()))
    return " ".join(parts) or str(key)


def _payload_field_value(value: object) -> str:
    if isinstance(value, (list, tuple, set)):
        try:
            return json.dumps(list(value), sort_keys=True, default=str)
        except Exception:
            return str(list(value))
    return str(value)


def _other_payload_subsection(key: str) -> str:
    if key in _OTHER_PAYLOAD_RECORD_RUN_CONTEXT_KEYS:
        return "Record & Run Context"
    if key in _OTHER_PAYLOAD_COORDINATES_GAIA_KEYS:
        return "Coordinates & Gaia"
    if key in _OTHER_PAYLOAD_PERIOD_FILTER_KEYS:
        return "Period & Filter Flags"
    if key.startswith("derived_"):
        return "Derived Feature Extras"
    if key.startswith("char_status_"):
        return "Enrichment Stage Status"
    if key.startswith("sed_"):
        return "SED Pipeline Status"
    if key.startswith(_OTHER_PAYLOAD_EXTERNAL_LC_PREFIXES):
        return "External LC Coverage"
    if key.startswith("ms_"):
        return "Multi-Survey Features"
    if (
        (key.startswith("apass_") and key.endswith("_err"))
        or (key.startswith("tmass_") and key.endswith("_err"))
        or key in _OTHER_PAYLOAD_WISE_ERROR_KEYS
    ):
        return "Photometric Error Columns"
    return "Miscellaneous"


def _other_payload_feature_rows(payload: dict, represented_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    represented_keys = {str(row.get("key") or "") for row in represented_rows}
    rows: list[dict[str, str]] = []
    if not isinstance(payload, dict):
        return rows
    for raw_key, value in sorted(payload.items(), key=lambda item: str(item[0])):
        key = str(raw_key)
        if key in represented_keys or key in _OTHER_PAYLOAD_CONTAINER_KEYS:
            continue
        if isinstance(value, dict) or not _payload_value_is_present(value):
            continue
        rows.append({
            "section": "Other Payload Fields",
            "source_group": "Payload",
            "label": _payload_field_label(key),
            "key": key,
            "value": _payload_field_value(value),
            "role": "other",
            "subsection": _other_payload_subsection(key),
        })
    rows.sort(key=lambda row: (_OTHER_PAYLOAD_SUBSECTION_INDEX.get(row["subsection"], 999), row["key"]))
    return rows


def _raw_feature_line(row: dict[str, str]) -> str:
    section = str(row.get("section") or "")
    subsection = str(row.get("subsection") or "")
    label = str(row.get("label") or "")
    key = str(row.get("key") or "")
    value = str(row.get("value") or "")
    if subsection:
        return f"{section} / {subsection} / {label} / {key} = {value}"
    return f"{section} / {label} / {key} = {value}"


def _render_all_features_lines(rows: list[dict[str, str]]) -> html.Div:
    return html.Div([
        html.Div([
            html.Span(row["label"], className="all-features-label"),
            html.Span(row["key"], className="all-features-key"),
            html.Span(row["value"], className="all-features-value"),
        ], className="all-features-line", title=_raw_feature_line(row))
        for row in rows
    ], className="all-features-lines")


def _render_other_payload_feature_lines(rows: list[dict[str, str]]) -> html.Div:
    rows_by_subsection: dict[str, list[dict[str, str]]] = {
        subsection: []
        for subsection in _OTHER_PAYLOAD_SUBSECTION_ORDER
    }
    for row in rows:
        subsection = str(row.get("subsection") or "Miscellaneous")
        rows_by_subsection.setdefault(subsection, []).append(row)

    subsection_nodes = []
    for subsection in _OTHER_PAYLOAD_SUBSECTION_ORDER:
        subsection_rows = rows_by_subsection.get(subsection, [])
        if not subsection_rows:
            continue
        subsection_rows = sorted(subsection_rows, key=lambda row: row["key"])
        subsection_nodes.append(html.Div([
            html.Div(f"{subsection} ({len(subsection_rows)})", className="all-features-subsection-title"),
            _render_all_features_lines(subsection_rows),
        ], className="all-features-subsection"))

    return html.Div(subsection_nodes, className="all-features-subsections")


def _render_all_features_plain_list(rows: list[dict[str, str]]) -> html.Details:
    """Render the exhaustive metadata/stat dump without spreadsheet chrome."""
    normalized = [
        {
            "section": str(row.get("section") or "Other"),
            "subsection": str(row.get("subsection") or ""),
            "label": str(row.get("label") or ""),
            "key": str(row.get("key") or ""),
            "value": str(row.get("value") or ""),
        }
        for row in rows
    ]
    copy_text = "\n".join(_raw_feature_line(row) for row in normalized)

    groups: dict[str, list[dict[str, str]]] = {}
    for row in normalized:
        groups.setdefault(row["section"], []).append(row)

    group_nodes = []
    for section, section_rows in groups.items():
        lines = (
            _render_other_payload_feature_lines(section_rows)
            if section == "Other Payload Fields"
            else _render_all_features_lines(section_rows)
        )
        group_nodes.append(html.Div([
            html.Div(f"{section} ({len(section_rows)})", className="all-features-group-title"),
            lines,
        ], className="all-features-group"))

    return html.Details([
        html.Summary(f"All Features ({len(normalized)})"),
        html.Div([
            html.Div([
                html.Button(
                    "Copy all",
                    type="button",
                    title="Copy all raw features",
                    className="metadata-copy-btn all-features-copy-btn",
                    **{
                        "aria-label": "Copy all raw features",
                        "data-copy-text": copy_text,
                    },
                ),
            ], className="all-features-copy-row"),
            html.Div(group_nodes, className="all-features-plain"),
        ], className="all-features-wrap"),
    ],
        id={"type": "meta-details", "group": "All Features"},
        open=False,
        className="review-feature-section all-features-details",
    )


def _render_metadata_review_layout(
    payload: dict,
    grouped: list,
    stat_rows: list[tuple[str, str]],
    feature_rows: list[dict[str, str]],
) -> list:
    """Render the streamlined candidate metadata panel."""
    output: list = []

    dipper_probability_card = _render_dipper_probability_card(payload)
    if dipper_probability_card is not None:
        output.append(dipper_probability_card)

    rows_by_section: dict[str, list[dict[str, str]]] = {section: [] for section in _REVIEW_SECTION_ORDER}
    for row in feature_rows:
        section = str(row.get("section") or "Advanced Metadata")
        rows_by_section.setdefault(section, []).append(row)

    dip_open = (
        bool(_truthy_display(payload.get("dip_significant")))
        or _numeric_payload_value(payload, "dipper_score") > 0
    )
    jump_open = (
        bool(_truthy_display(payload.get("jump_significant")))
        or _numeric_payload_value(payload, "jumper_score") > 0
        or _numeric_payload_value(payload, "jump_count") > 0
    )
    open_defaults = {
        "Review Summary": True,
        "Dip Evidence": dip_open,
        "Jump Evidence": jump_open,
        "Coverage & Photometry": True,
        "Catalog & Vetting": False,
        "Classification & Environment": False,
        "Advanced Metadata": False,
    }

    for section in _REVIEW_SECTION_ORDER:
        rendered = _render_feature_section(
            section,
            rows_by_section.get(section, []),
            open_default=open_defaults.get(section, False),
        )
        if rendered is not None:
            output.append(rendered)

    stat_cards = _render_stat_cards(stat_rows)
    if stat_cards:
        output.append(html.Details([
            html.Summary(f"Advanced Stats ({len(stat_rows)})"),
            html.Div(stat_cards, className="advanced-stats-wrap"),
        ],
            id={"type": "meta-details", "group": "Advanced Stats"},
            open=False,
            className="review-feature-section advanced-stats-details",
        ))

    all_rows = list(feature_rows) + _stat_feature_rows(stat_rows)
    all_rows.extend(_other_payload_feature_rows(payload, all_rows))
    output.append(_render_all_features_plain_list(all_rows))
    return output


def _render_stat_cards(stat_rows: list[tuple[str, str]]) -> list:
    """Render stats as grouped collapsible sections with readable labels."""
    if not stat_rows:
        return []

    label_overrides = {
        "stats_jd_start": "JD start (day)",
        "stats_jd_end": "JD end (day)",
        "stats_time_span_days": "Time span (days)",
        "stats_n_unique_nights": "Unique nights",
        "stats_duty_cycle_fraction": "Duty cycle",
        "stats_file_points_total": "Points total",
        "stats_file_points_kept_after_filter": "Points kept after filter",
        "stats_cadence_mean_dt_days": r"Cadence mean $\Delta t$ (days)",
        "stats_cadence_median_dt_days": r"Cadence median $\Delta t$ (days)",
        "stats_cadence_p05_dt_days": r"Cadence $P_{05}$ $\Delta t$ (days)",
        "stats_cadence_p95_dt_days": r"Cadence $P_{95}$ $\Delta t$ (days)",
        "stats_photometry_band_mode": "Photometry bands used",
        "stats_photometry_band_alignment": "Band alignment",
        "stats_photometry_g_points": "g-band points",
        "stats_photometry_v_points": "V-band points",
        "stats_photometry_v_minus_g_offset_mag": "V-g offset applied (mag)",
        "stats_photometry_robust_sigma_mag": r"Robust $\sigma_m$ (mag)",
        "stats_photometry_std_mag": r"$\sigma_m$ (mag)",
        "stats_photometry_IQR_mag": r"IQR($m$) (mag)",
        "stats_photometry_mean_mag": r"Mean($m$) (mag)",
        "stats_photometry_median_mag": r"Median($m$) (mag)",
        "stats_photometry_weighted_mean_mag": r"Weighted mean($m$) (mag)",
        "stats_photometry_weighted_mean_sem": r"SEM($\bar{m}_w$) (mag)",
        "stats_photometry_weighted_std_mag": r"Weighted $\sigma_m$ (mag)",
        "stats_photometry_p05_mag": r"$P_{05}(m)$ (mag)",
        "stats_photometry_p16_mag": r"$P_{16}(m)$ (mag)",
        "stats_photometry_p84_mag": r"$P_{84}(m)$ (mag)",
        "stats_photometry_p95_mag": r"$P_{95}(m)$ (mag)",
        "stats_clipped_mean_mag_3sigma_about_median": r"Clipped mean (3$\sigma$ about median) (mag)",
        "stats_clipped_mean_mag_3sigma_about_median_g": r"Clipped mean (3$\sigma$ about median, g) (mag)",
        "stats_clipped_mean_mag_3sigma_about_median_vband": r"Clipped mean (3$\sigma$ about median, V) (mag)",
        "stats_clipped_std_mag_3sigma_about_median": r"Clipped std (3$\sigma$ about median) (mag)",
        "stats_n_outliers_removed_robust_3sigma": r"Outliers removed (robust 3$\sigma$)",
        "stats_error_and_snr_stats_error_mean": "Error mean (mag)",
        "stats_error_and_snr_stats_error_median": "Error median (mag)",
        "stats_error_and_snr_stats_error_p05": r"Error $P_{05}$ (mag)",
        "stats_error_and_snr_stats_error_p95": r"Error $P_{95}$ (mag)",
        "stats_error_and_snr_stats_snr_median": "SNR median",
        "stats_error_and_snr_stats_snr_p05": r"SNR $P_{05}$",
        "stats_error_and_snr_stats_snr_p95": r"SNR $P_{95}$",
        "stats_variability_reduced_chi2_vs_constant": r"Reduced $\chi^2$ vs constant",
        "stats_variability_von_neumann_ratio": r"Inverse von Neumann ratio $1/\eta$",
        "stats_variability_roms": "RoMS",
        "stats_variability_sokolovsky_v": r"Sokolovsky peak-to-peak $v$",
        "stats_variability_sokolovsky_v_g": r"Sokolovsky peak-to-peak $v$ (g)",
        "stats_variability_sokolovsky_v_vband": r"Sokolovsky peak-to-peak $v$ (V)",
        "stats_variability_lag1_autocorr": r"Lag-1 $\rho$",
        "stats_variability_stetson_I": r"Stetson $I$",
        "stats_variability_stetson_J": r"Stetson $J$",
        "stats_variability_stetson_K": r"Stetson $K$",
        "stats_variability_stetson_L": r"Stetson $L$",
        "stats_variability_stetson_J_time": r"Stetson $J_\mathrm{time}$",
        "stats_variability_stetson_L_time": r"Stetson $L_\mathrm{time}$",
        "stats_variability_flux_asymmetry_m": r"Flux asymmetry $M$",
        "stats_variability_quasi_periodicity_q": r"Phase-template quasi-periodicity $Q$",
        "stats_variability_quasi_periodicity_method": "Q method",
        "stats_variability_quasi_periodicity_n_points": "Q point count",
        "stats_variability_quasi_periodicity_n_bins": "Q phase bins",
        "stats_variability_quasi_periodicity_populated_bins": "Q populated bins",
        "stats_variability_quasi_periodicity_bin_coverage": "Q bin coverage",
        "stats_variability_quasi_periodicity_smooth_window_bins": "Q smoothing bins",
        "stats_variability_quasi_periodicity_template_amplitude": "Q template amplitude (mag)",
        "stats_variability_quasi_periodicity_raw_scatter": "Q raw scatter (mag)",
        "stats_variability_quasi_periodicity_resid_scatter": "Q residual scatter (mag)",
        "stats_variability_quasi_periodicity_scatter_ratio": "Q residual/raw scatter",
        "stats_variability_quasi_periodicity_status": "Q status",
        "stats_variability_periodic_feature_period_days": "Q feature period (days)",
        "stats_variability_periodic_feature_period_source": "Q feature period source",
        "stats_variability_string_length_resid_total": "String length total (mag)",
        "stats_variability_string_length_resid_mean_step": "String length mean step (mag)",
        "stats_variability_string_length_resid_n_steps": "String length n steps",
        "stats_variability_lomb_scargle_best_period_days": "Lomb-Scargle best period (days)",
        "stats_variability_lomb_scargle_peak_power": "Lomb-Scargle peak power",
        "stats_variability_lomb_scargle_fap": "Lomb-Scargle FAP",
        "stats_lafler_kinman_t_time": r"Lafler-Kinman $T(t)$",
        "stats_lafler_kinman_t_phase": r"Lafler-Kinman $T(\phi|P)$",
        "stats_lafler_kinman_delta": r"Lafler-Kinman $\delta$",
        "stats_trend_slope_mag_per_day": r"$\mathrm{d}m/\mathrm{d}t$ (mag/day)",
        "stats_trend_slope_mag_per_year": r"$\mathrm{d}m/\mathrm{d}t$ (mag/year)",
        "stats_trend_r2": r"Trend $R^2$",
        "stats_gp_drw_sigma": r"GP-DRW $\sigma$ (mag)",
        "stats_gp_drw_tau": r"GP-DRW $\tau$",
        "stats_iar_phi": r"IAR $\phi$",
        "stats_sf_ml_amplitude": "SF-ML amplitude (mag)",
        "stats_sf_ml_gamma": r"SF-ML $\gamma$",
        "stats_psi_cs": r"Psi CS ($\psi_{\mathrm{CS}}$)",
        "stats_psi_eta": r"Psi eta ($\psi_{\eta}$)",
        "stats_con": "Con statistic",
        "stats_intrinsic_sigma_mag": r"Intrinsic $\sigma$ (mag)",
        "stats_excess_var": r"Intrinsic $\sigma$ (mag)",
        "stats_amplitude": "Amplitude (mag)",
        "stats_first_mag": r"First $m$ (mag)",
        "stats_max_slope": "Max slope (mag/day)",
        "stats_median_abs_dev": r"Median abs dev (mag)",
        "stats_gskew": "g-skew",
        "stats_meanvariance": "Mean/variance",
        "stats_median_brp": "Median BRP",
        "stats_constancy_p_value": "Constancy p-value",
        "stats_pvar": "Constancy p-value",
        "stats_q31": r"$Q_{31}$ (mag)",
        "stats_rcs": "RCS",
        "stats_autocor_length": "Autocorrelation length",
        "stats_delta_mag_fid": r"$\Delta m_{\mathrm{fid}}$ (mag)",
        "stats_beyond_1_std": "Beyond 1 std",
        "stats_small_kurtosis": "Small kurtosis",
        "stats_pair_slope_trend": "Pair slope trend",
        "stats_ahl_ratio": "AHL bright/faint ratio",
        "stats_eb_rminima": "EB minimum depth ratio",
        "stats_eb_primary_min_depth": "EB primary minimum depth",
        "stats_eb_secondary_min_depth": "EB secondary minimum depth",
        "stats_harmonics_mse": r"$\mathrm{MSE}$ ($\mathrm{mag}^2$)",
        "stats_harmonics_order": "Recommended harmonic order",
        "stats_harmonics_period": "Adopted period (d)",
        "stats_harmonics_a0": r"Zero-point $A_0$ (mag)",
        "stats_harmonics_model_amplitude": "Model amplitude (mag)",
        "stats_harmonics_reduced_chi2": r"Reduced $\chi^2$",
        "stats_mhps_pn_flag": "MHPS PN flag",
        "stats_mhps_non_zero": "MHPS non-zero count",
        "stats_asassn_field_key": "ASAS-SN field",
        "stats_asassn_fields": "ASAS-SN fields",
        "stats_asassn_field_count": "ASAS-SN field count",
        "stats_asassn_field_key_fraction": "ASAS-SN field fraction",
        "stats_camera_name_key": "Camera name",
        "stats_camera_names": "Camera names",
        "stats_camera_name_count": "Camera name count",
        "stats_camera_name_key_fraction": "Camera name fraction",
        "ltv_median": "Median (mag)",
        "ltv_median_err": "Median err proxy (mag)",
        "time_span_days": "Time span (days)",
        "n_unique_nights": "Unique nights",
        "ltv_vg_has_v": "Has V band",
        "ltv_vg_overlap_days": "V/g overlap (days)",
        "ltv_vg_overlap_fraction": "V/g overlap fraction",
        "filtered_cams": "Filtered cameras",
        "derived_bp_rp": r"Derived BP$-$RP",
        "derived_j_k": r"Derived J$-$Ks",
        "derived_mrp": r"Derived $M_{RP}$",
        "derived_mks": r"Derived $M_{Ks}$",
        "derived_wrp": r"Derived $W_{RP}$",
        "derived_wjk": r"Derived $W_{JK}$",
        "derived_harmonics_r32": r"Amplitude ratio $R_{32}$",
        "derived_harmonics_r42": r"Amplitude ratio $R_{42}$",
        "derived_harmonics_r43": r"Amplitude ratio $R_{43}$",
        "derived_harmonics_a4_a2": r"Fourier coefficient ratio $a_4/a_2$",
        "derived_harmonics_b4_b2": r"Fourier coefficient ratio $b_4/b_2$",
    }

    alerce_feature_keys = {
        "stats_amplitude",
        "stats_beyond_1_std",
        "stats_con",
        "stats_delta_mag_fid",
        "stats_intrinsic_sigma_mag",
        "stats_excess_var",
        "stats_first_mag",
        "stats_gskew",
        "stats_max_slope",
        "stats_meanvariance",
        "stats_median_abs_dev",
        "stats_median_brp",
        "stats_percent_amplitude",
        "stats_ahl_ratio",
        "stats_q31",
        "stats_skew",
        "stats_small_kurtosis",
        "stats_constancy_p_value",
        "stats_pvar",
        "stats_anderson_darling",
        "stats_pair_slope_trend",
        "stats_rcs",
        "stats_autocor_length",
    }

    group_order = [
        "LTV Summary",
        "LTV Trend",
        "LTV Seasons",
        "LTV Stochastic",
        "Coverage & Cadence",
        "Photometry & SNR",
        "Periodicity",
        "Variability",
        "Trend",
        "Harmonics",
        "Derived Features",
        "Stochastic Models",
        "MHPS / Structure Function",
        "ALeRCE Features",
        "Camera Diagnostics",
        "Other",
    ]

    def _stat_group(key: str) -> str:
        if key == "filtered_cams":
            return "Camera Diagnostics"
        if key.startswith("stats_asassn_field_") or key.startswith("stats_camera_name_"):
            return "Camera Diagnostics"
        if key.startswith("ltv_stoch_"):
            return "LTV Stochastic"
        if key.startswith("ltv_season_") or key.startswith("ltv_leave1out_"):
            return "LTV Seasons"
        if key.startswith("ltv_trend_") or key in {
            "ltv_slope", "ltv_slope_quad", "ltv_max_diff", "ltv_coeff1", "ltv_coeff2"
        }:
            return "LTV Trend"
        if key.startswith("ltv_"):
            return "LTV Summary"
        if key.startswith("stats_file_points_") or key in {
            "stats_jd_start", "stats_jd_end", "stats_time_span_days",
            "stats_n_unique_nights", "stats_duty_cycle_fraction",
        } or key.startswith("stats_cadence_"):
            return "Coverage & Cadence"
        if key.startswith("stats_photometry_") or key.startswith("stats_error_and_snr_stats_") or key.startswith("stats_clipped_") or key == "stats_n_outliers_removed_robust_3sigma":
            return "Photometry & SNR"
        if (
            key.startswith("stats_variability_lomb_scargle_")
            or key.startswith("stats_psi_")
            or key.startswith("stats_lafler_kinman_")
            or key.startswith("stats_window_alias_")
        ):
            return "Periodicity"
        if key.startswith("stats_variability_"):
            return "Variability"
        if key.startswith("stats_trend_"):
            return "Trend"
        if key.startswith("stats_harmonics_"):
            return "Harmonics"
        if key.startswith("stats_eb_"):
            return "Harmonics"
        if key.startswith("derived_"):
            return "Derived Features"
        if key.startswith("stats_gp_drw_") or key.startswith("stats_iar_"):
            return "Stochastic Models"
        if key.startswith("stats_mhps_") or key.startswith("stats_sf_ml_"):
            return "MHPS / Structure Function"
        if key in alerce_feature_keys:
            return "ALeRCE Features"
        return "Other"

    def _fallback_label(key: str) -> str:
        if key.startswith("stats_"):
            raw = key[6:]
        elif key.startswith("ltv_"):
            raw = key[4:]
        else:
            raw = key
        token_map = {
            "jd": "JD",
            "snr": "SNR",
            "iqr": "IQR",
            "std": "Std",
            "gp": "GP",
            "drw": "DRW",
            "iar": "IAR",
            "mhps": "MHPS",
            "rcs": "RCS",
            "fap": "FAP",
            "bic": "BIC",
            "ls": "LS",
            "vg": "V/g",
            "ml": "ML",
            "chi2": r"$\chi^2$",
            "r2": r"$R^2$",
            "sigma": r"$\sigma$",
            "tau": r"$\tau$",
            "phi": r"$\phi$",
            "rho": r"$\rho$",
            "gamma": r"$\gamma$",
            "eta": r"$\eta$",
            "psi": r"$\psi$",
        }
        parts: list[str] = []
        for tok in [t for t in raw.split("_") if t]:
            lower = tok.lower()
            p_match = re.fullmatch(r"p(\d{2})", lower)
            if p_match:
                parts.append(rf"$P_{{{p_match.group(1)}}}$")
            elif lower in token_map:
                parts.append(token_map[lower])
            elif lower.isdigit():
                parts.append(lower)
            else:
                parts.append(lower.capitalize())
        return " ".join(parts)

    def _stat_label(key: str) -> str:
        if key in label_overrides:
            return label_overrides[key]
        mag_match = re.fullmatch(r"stats_harmonics_mag_(\d+)", key)
        if mag_match:
            n = mag_match.group(1)
            return rf"Amplitude $A_{{{n}}}$ (mag)"
        coeff_match = re.fullmatch(r"stats_harmonics_([ab])(\d+)", key)
        if coeff_match:
            coeff, n = coeff_match.groups()
            return rf"Fourier coefficient ${coeff}_{{{n}}}$"
        ratio_match = re.fullmatch(r"stats_harmonics_r(\d+)1", key)
        if ratio_match:
            n = ratio_match.group(1)
            return rf"Amplitude ratio $R_{{{n}1}}$"
        phase_match = re.fullmatch(r"stats_harmonics_phase_(\d+)", key)
        if phase_match:
            n = phase_match.group(1)
            return rf"Phase combination $\phi_{{{n}1}}$ (rad)"
        return _fallback_label(key)

    grouped: dict[str, list[tuple[str, str]]] = {name: [] for name in group_order}
    for key_raw, value in stat_rows:
        key = str(key_raw)
        grouped.setdefault(_stat_group(key), []).append((markdown_literal_unit_label(_stat_label(key)), str(value)))

    sections = []
    for group_name in group_order:
        rows = grouped.get(group_name, [])
        if not rows:
            continue
        field_divs = [
            html.Div([
                dcc.Markdown(label, className='meta-field-label stat-field-label', mathjax=True),
                _copyable_math_value(value),
            ], className='meta-field-row')
            for label, value in rows
        ]
        sections.append(
            html.Details(
                [html.Summary(f"{group_name} ({len(rows)})"), html.Div(field_divs, className='meta-grid')],
                open=group_name in {"Coverage & Cadence", "Photometry & SNR"},
                className='stats-section',
            )
        )

    total_rows = sum(len(grouped.get(name, [])) for name in group_order)
    return [html.Details(
        [html.Summary(f"Stats ({total_rows})"), html.Div(sections, className='metadata-sections stats-sections-grid')],
        open=True,
        className='stats-details',
    )]


def _render_plot_status_panel(status: str, message: str, warnings: list[str] | None) -> html.Div:
    """Render native plot status and warnings panel."""
    warnings = [str(w) for w in (warnings or []) if str(w).strip()]
    cls = 'plot-status'
    if status in {'missing-file', 'missing-columns', 'empty-after-filter', 'empty-camera-selection', 'error'}:
        cls += ' error'
    elif warnings:
        cls += ' warn'

    base_message = str(message or '').strip()
    if not base_message:
        base_message = 'Native interactive plot active.'

    headline = html.Div(base_message, className='status-line')
    if not warnings:
        return html.Div([headline], className=cls)

    warning_items = [html.Li(w) for w in warnings[:8]]
    if len(warnings) > 8:
        warning_items.append(html.Li(f"...and {len(warnings) - 8} more"))

    details = html.Details([
        html.Summary(f"{len(warnings)} warning(s)"),
        html.Ul(warning_items),
    ])
    return html.Div([headline, details], className=cls)


def _render_camera_diag_panel(camera_diagnostics: dict[str, list[str]], filtered_values: list[str] | None) -> list:
    """Render explainable camera filtering tags."""
    if not filtered_values:
        return []

    chips = []
    for cam in sorted(filtered_values):
        reasons = camera_diagnostics.get(str(cam), [])
        label = f"Cam {cam}: {','.join(reasons) if reasons else 'unknown'}"
        chips.append(html.Div(label, className='item'))
    return chips


def _run_params_path_for_plot_dir(plot_dir: str | None) -> Path | None:
    """Get run_params.json path from current plot directory."""
    if not plot_dir:
        return None
    candidate = Path(plot_dir).resolve().parent / "run_params.json"
    return candidate if candidate.exists() else None


def _derive_defaults_from_run_params(run_params: dict | None) -> tuple[str, list[str]]:
    """Derive initial preset and overlays from run config."""
    if not run_params:
        preset = 'Diagnostics'
        return preset, list(PLOT_PRESETS[preset]['overlays'])

    run_filter = bool(run_params.get('run_filter', True))
    run_postprocess = bool(run_params.get('run_postprocess', True))
    min_bf = float(run_params.get('min_bayes_factor', 0.0) or 0.0)
    if (not run_filter) and (not run_postprocess):
        preset = 'Clean'
    elif min_bf >= 12:
        preset = 'Full'
    else:
        preset = 'Diagnostics'

    overlays = set(PLOT_PRESETS[preset]['overlays'])
    if bool(run_params.get('skip_camera_median', False)):
        overlays.discard('filter_bad_cameras')
    if not bool(run_params.get('run_postprocess', True)):
        overlays.discard('diagnostics')
        overlays.discard('confidence')
    return preset, sorted(list(overlays))


def _run_config_rows(run_params: dict) -> list[tuple[str, str]]:
    """Compact rows for run config panel."""
    rows: list[tuple[str, str]] = []
    for label, key in (
        ('Stage', 'stage'),
        ('Baseline', 'baseline_func'),
        ('Baseline S0', 'baseline_s0'),
        ('Baseline w0', 'baseline_w0'),
        ('Baseline q', 'baseline_q'),
        ('Baseline jitter', 'baseline_jitter'),
        ('Baseline sigma floor', 'baseline_sigma_floor'),
        ('Trigger mode', 'trigger_mode'),
        ('Workers', 'workers'),
        ('Batch size', 'batch_size'),
        ('Min Bayes factor', 'min_bayes_factor'),
        ('LogBF dip thr', 'logbf_threshold_dip'),
        ('LogBF jump thr', 'logbf_threshold_jump'),
        ('Significance thr', 'significance_threshold'),
        ('Clean err abs', 'clean_max_error_absolute'),
        ('Clean err sigma', 'clean_max_error_sigma'),
        ('Bad cam scatter', 'bad_camera_scatter_ratio'),
    ):
        val = run_params.get(key)
        if val is None:
            continue
        rows.append((label, str(val)))
    return rows


def _run_config_mismatch_warnings(run_params: dict | None, overlays: set[str]) -> list[str]:
    """Warnings for GUI/view assumptions mismatching run config."""
    if not run_params:
        return ["run_params.json missing; native plot uses fallback defaults."]

    warns: list[str] = []
    expected_filter_bad = not bool(run_params.get('skip_camera_median', False))
    if ('filter_bad_cameras' in overlays) != expected_filter_bad:
        warns.append(
            f"Bad-camera filter toggle differs from run config (expected {'on' if expected_filter_bad else 'off'})."
        )

    if run_params.get('baseline_func') is None:
        warns.append("baseline_func missing in run_params; baseline defaults may differ from original run.")
    if run_params.get('clean_max_error_absolute') is None or run_params.get('clean_max_error_sigma') is None:
        warns.append("cleaning thresholds missing in run_params; fallback cleaning defaults are active.")
    return warns


def _path_is_under(path: Path | None, root: Path | None) -> bool:
    """Return True when *path* is located under *root*."""
    if path is None or root is None:
        return False
    path = Path(path).expanduser()
    root = Path(root).expanduser()
    try:
        path.relative_to(root)
        return True
    except ValueError:
        pass
    except Exception:
        pass
    try:
        if not path.exists() or not root.exists():
            return False
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _paths_equal_existing(left: object, right: object) -> bool:
    """Compare paths without resolving nonexistent stale absolute paths."""
    if left in (None, "") or right in (None, ""):
        return False
    try:
        left_path = Path(str(left)).expanduser()
        right_path = Path(str(right)).expanduser()
    except Exception:
        return False
    if str(left_path) == str(right_path):
        return True
    try:
        if not left_path.exists() or not right_path.exists():
            return False
        return left_path.resolve() == right_path.resolve()
    except Exception:
        return False


def _baseline_provenance_warning(
    payload: dict,
    *,
    plot_dir: Path | None,
    run_params: dict | None,
    stored_lc_path: object = None,
    source_path: object = None,
) -> str | None:
    """Describe when the displayed baseline is not provenance-matched to a pipeline run."""
    lc_path = resolve_lightcurve_path(payload, plot_dir)
    if lc_path is None:
        return None

    if not run_params:
        return "Baseline is recomputed live in review with fallback defaults; no run_params.json is loaded."

    source_text = str(source_path or '').strip()
    if source_text.startswith('fetch://'):
        return "Baseline is recomputed from the imported SkyPatrol light curve using the current run settings; it is not a saved pipeline baseline."

    plot_run_dir = plot_dir.parent if plot_dir is not None else None
    source_run_dir = _run_dir_from_source_path(source_path)

    if _path_is_under(lc_path, plot_run_dir) or _path_is_under(lc_path, source_run_dir):
        return None

    cluster_path = str(payload.get('path') or '').strip()
    if cluster_path and _paths_equal_existing(cluster_path, lc_path):
        return None

    stored_lc_text = str(stored_lc_path or '').strip()
    if stored_lc_text and _paths_equal_existing(stored_lc_text, lc_path):
        return "Baseline is recomputed from the local review copy of this light curve; it may differ from the original pipeline baseline source."

    if lc_path.suffix.lower() == '.csv':
        return "Baseline is recomputed from a local CSV light curve in review; it may not match the original pipeline baseline source."

    return "Baseline is recomputed from the currently resolved local light curve; it may differ from the original pipeline baseline source."


def _render_run_config_panel(run_params: dict | None, run_params_path: Path | None, warnings: list[str]) -> list:
    """Render run config as simple meta-field rows (same style as stats/metadata)."""
    rows: list[tuple[str, str]] = [
        ('Status', 'Loaded' if run_params else 'Missing'),
        ('Path', str(run_params_path) if run_params_path else 'not found'),
    ]
    rows.extend(_run_config_rows(run_params or {}))
    if warnings:
        rows.append(('Warnings', ' | '.join(warnings)))

    field_divs = [
        html.Div([
            html.Span(label, className='meta-field-label'),
            html.Span(str(value), className='meta-field-value'),
        ], className='meta-field-row')
        for label, value in rows
    ]
    return [html.Div(field_divs, className='meta-grid')]


def _render_repro_badge(run_params: dict | None, warnings: list[str]) -> html.Span:
    """Render reproducibility status badge."""
    if (run_params is None) or warnings:
        text = 'Repro: fallback/defaults'
        cls = 'repro-badge warn'
    else:
        text = 'Repro: exact run params'
        cls = 'repro-badge'
    return html.Span(text, className=cls)


def _format_queue_count(value: object) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "0"


def _render_queue_filter_provenance_panel(queue_data: dict | None) -> list:
    """Render queue attrition summary for active sidebar filters."""
    data = dict(queue_data or {})
    scope_size = int(data.get("scope_size") or 0)
    visible_size = int(data.get("queue_size") or 0)
    filtered_out = int(data.get("filtered_out_count") or max(scope_size - visible_size, 0))
    active_filters = list(data.get("filter_provenance") or [])

    summary_rows = [
        ("Scoped queue", _format_queue_count(scope_size)),
        ("Visible now", _format_queue_count(visible_size)),
        ("Filtered out", _format_queue_count(filtered_out)),
        ("Active sidebar filters", _format_queue_count(len(active_filters))),
    ]

    summary = html.Div([
        html.Div([
            html.Span(label, className='meta-field-label'),
            html.Span(str(value), className='meta-field-value'),
        ], className='meta-field-row')
        for label, value in summary_rows
    ], className='meta-grid')

    if active_filters:
        details = html.Ul([
            html.Li(
                f"{item.get('label', 'filter')}: "
                f"{_format_queue_count(item.get('filtered_count', 0))} filtered out "
                f"({_format_queue_count(item.get('remaining_count', 0))} remain)"
            )
            for item in active_filters
        ], className='queue-provenance-list')
        note = html.Div(
            "Per-filter counts are relative to the full scoped queue. "
            "Filters can overlap, so these counts do not sum to the total filtered-out rows.",
            className='queue-provenance-note',
        )
    else:
        details = html.Div(
            "No sidebar filters are active. The visible queue matches the full scoped queue.",
            className='queue-provenance-note',
        )
        note = None

    children: list = [summary, details]
    if note is not None:
        children.append(note)
    return [html.Div(children, className='queue-provenance-panel')]


_METADATA_EXTRA_GROUPS = (
    "Triage Summary",
    "Vetting",
    "Period Consensus",
    "External Follow-up",
    "Stellar Parameters",
    "Photometry",
    "Environment",
    "YSO / Classification",
)

_METADATA_CATALOG_GROUPS = (
    "Vetting",
    "Period Consensus",
    "External Follow-up",
    "Stellar Parameters",
    "Photometry",
    "Environment",
)


def _summarize_group_names(group_names: list[str], max_items: int = 2) -> str:
    names = [str(n) for n in group_names if str(n).strip()]
    if not names:
        return "none"
    if len(names) <= max_items:
        return ", ".join(names)
    return f"{', '.join(names[:max_items])} +{len(names) - max_items}"


def _render_metadata_health(grouped: list[tuple[str, list[tuple[str, object]]]] | None, *, context_msg: str | None = None) -> html.Div:
    """Render compact metadata-enrichment status for current candidate."""
    if context_msg:
        return html.Div([
            html.Span("Base only", className='chip'),
            html.Span(str(context_msg), className='detail'),
        ], className='metadata-health metadata-health-base')

    grouped_map = {name: items for name, items in (grouped or [])}
    extra_present = [name for name in _METADATA_EXTRA_GROUPS if grouped_map.get(name)]
    catalog_present = [name for name in _METADATA_CATALOG_GROUPS if grouped_map.get(name)]
    extra_fields = int(sum(len(grouped_map.get(name) or []) for name in extra_present))

    if not extra_present:
        return html.Div([
            html.Span("Base only", className='chip'),
            html.Span(
                "No crossmatch/classification/catalog metadata fields are present for this candidate.",
                className='detail',
            ),
        ], className='metadata-health metadata-health-base')

    if catalog_present:
        detail = (
            f"Catalog metadata present in {_summarize_group_names(catalog_present)} "
            f"({extra_fields} extra fields total)."
        )
        return html.Div([
            html.Span("Catalog enriched", className='chip'),
            html.Span(detail, className='detail'),
        ], className='metadata-health metadata-health-enriched')

    extra_label = _summarize_group_names(extra_present)
    return html.Div([
        html.Span("Partial", className='chip'),
        html.Span(
            f"Only {extra_label} metadata is present ({extra_fields} extra fields); no catalog stellar/photometric enrichment.",
            className='detail',
        ),
    ], className='metadata-health metadata-health-partial')


def _render_vetting_banner(payload: dict | None, radius_arcsec: float = 30.0) -> html.Div:
    """Render a vetting status panel with source cards above the metadata grid."""
    if not payload:
        return html.Div("Not vetted", className='vetting-banner-empty')

    def _ok(v) -> bool:
        """True if v is a non-empty, non-NaN string."""
        return bool(v) and str(v).strip().lower() not in ('nan', '<na>')

    known = has_known_catalog_evidence(payload)
    if not known and not has_catalog_vetting_context(payload):
        return html.Div("Not vetted", className='vetting-banner-empty')

    banner_state = 'known' if known else 'new'

    # Status header
    header_text = "KNOWN VARIABLE" if known else "POTENTIALLY NEW"

    cards = []

    def _label(text: str) -> html.Span:
        return html.Span(text, className='vetting-banner-label')

    def _value(text: str, *, hit: bool = False, title: str | None = None) -> html.Span:
        cls = 'vetting-banner-value'
        if hit:
            cls += f' vetting-banner-hit {banner_state}'
        return html.Span(text, className=cls, title=title)

    def _cell(left: str, right: str, *, hit: bool = False, title: str | None = None) -> html.Div:
        cell_class = 'vetting-banner-cell'
        if hit:
            cell_class += f' hit {banner_state}'
        return html.Div([
            _label(left),
            _value(right, hit=hit, title=title),
        ], className=cell_class)

    def _rich_cell(left: str, children, *, hit: bool = False) -> html.Div:
        cell_class = 'vetting-banner-cell vetting-banner-rich-cell'
        if hit:
            cell_class += f' hit {banner_state}'
        return html.Div([
            _label(left),
            html.Div(children, className='vetting-banner-value vetting-banner-rich-value'),
        ], className=cell_class)

    def _catalog_class_short_label(column: str, value: object) -> str:
        resolved = resolve_catalog_class(column, value)
        if not resolved.value:
            return ""
        suffix = f" [{resolved.source}]" if resolved.source else ""
        if suffix and resolved.label.endswith(suffix):
            return resolved.label[: -len(suffix)]
        return resolved.label

    # SIMBAD cell
    simbad_id = payload.get('simbad_main_id')
    simbad_otype = payload.get('simbad_otype')
    if _ok(simbad_id) or _ok(simbad_otype):
        simbad_hit = is_known_variable_type_value('simbad_otype', simbad_otype)
        refs = payload.get('simbad_nbref')
        parts = []
        if _ok(simbad_otype):
            parts.append(_catalog_class_short_label('simbad_otype', simbad_otype))
        if _ok(simbad_id):
            parts.append(str(simbad_id))
        if refs:
            parts.append(f"({refs} refs)")
        cards.append(_cell("SIMBAD", " \u00b7 ".join(parts), hit=simbad_hit, title=str(simbad_id or '')))

    # VSX cell
    vsx_cls = payload.get('vsx_class')
    if _ok(vsx_cls):
        vsx_sep = payload.get('vsx_sep_arcsec')
        sep_str = f" ({vsx_sep:.1f}\")" if vsx_sep and not pd.isna(vsx_sep) else ""
        vsx_p = payload.get('vsx_period')
        p_str = f", P={vsx_p:.4f}d" if vsx_p and not pd.isna(vsx_p) else ""
        cards.append(_cell("VSX", f"{vsx_cls}{p_str}{sep_str}", hit=True))

    def _payload_float(*keys: str) -> float | None:
        for key in keys:
            try:
                value = payload.get(key)
                if value is None or pd.isna(value):
                    continue
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _format_vsx_neighbor(neighbor: VsxNeighbor):
        parts = [f"{neighbor.sep_arcsec:.1f}\""]
        if neighbor.name:
            parts.append(neighbor.name)
        if neighbor.type_label:
            parts.append(neighbor.type_label)
        if neighbor.period_days is not None:
            parts.append(f"P={neighbor.period_days:.5g} d")
        label = " \u00b7 ".join(parts)
        if neighbor.url:
            return html.A(
                label,
                href=neighbor.url,
                target='_blank',
                rel='noopener noreferrer',
                className='vetting-banner-nearby-vsx-link',
                title=neighbor.url,
            )
        return html.Span(label, className='vetting-banner-nearby-vsx-text')

    ra_for_vsx = _payload_float('ra', 'ra_deg')
    dec_for_vsx = _payload_float('dec', 'dec_deg')
    if ra_for_vsx is not None and dec_for_vsx is not None:
        neighbors = find_nearby_vsx(ra_for_vsx, dec_for_vsx, limit=3, radius_arcsec=radius_arcsec)
        if neighbors:
            label = "Nearby VSX"
            if not _ok(vsx_cls):
                label = "Nearby VSX (live)"
            cards.append(_rich_cell(
                label,
                [
                    html.Div(
                        _format_vsx_neighbor(neighbor),
                        className='vetting-banner-nearby-vsx-row',
                    )
                    for neighbor in neighbors
                ],
            ))

    # Gaia variability cell
    gaia_cls = payload.get('gaia_var_class')
    if _ok(gaia_cls):
        score = payload.get('gaia_var_score')
        score_str = f" ({score:.2f})" if score and not pd.isna(score) else ""
        cards.append(_cell("Gaia DR3", f"{gaia_cls}{score_str}", hit=True))

    # Gaia EB evidence cell.  The level is based on distinct evidence families;
    # Gaia's NSS EclipsingBinary copy of the photometric period counts only once.
    eb_period = payload.get('gaia_eb_period')
    eb_level = payload.get('gaia_eb_evidence_level')
    eb_families = payload.get('gaia_binary_evidence_families')
    has_eb_level = _ok(eb_level) and str(eb_level) != 'none'
    if (eb_period and not pd.isna(eb_period)) or has_eb_level:
        details = []
        if has_eb_level:
            details.append(str(eb_level).replace('_', ' '))
        if eb_period and not pd.isna(eb_period):
            details.append(f"P={eb_period:.4f} d")
        if _ok(eb_families):
            details.append(str(eb_families).replace(',', ' + '))
        cards.append(_cell("Gaia EB", " · ".join(details), hit=str(eb_level) not in {'', 'none'}))

    binary_level = payload.get('gaia_binary_evidence_level')
    nss_types = payload.get('gaia_nss_solution_types')
    if _ok(binary_level) and str(binary_level) != 'none' and not _ok(eb_level):
        value = str(binary_level).replace('_', ' ')
        if _ok(nss_types):
            value += f" · NSS {nss_types}"
        cards.append(_cell("Gaia binary", value, hit=True))

    # ASAS-SN cell
    asassn_type = payload.get('asassn_var_type')
    if _ok(asassn_type):
        period = payload.get('asassn_var_period')
        p_str = f" P={period:.4f}d" if period and not pd.isna(period) else ""
        cards.append(_cell("ASAS-SN", f"{asassn_type}{p_str}", hit=True))

    # Microlensing catalog cell
    if _coerce_bool(payload.get('microlens_match')):
        ml_catalog = payload.get('microlens_catalog')
        ml_name = payload.get('microlens_name')
        ml_te = payload.get('microlens_te_days')
        ml_sep = payload.get('microlens_sep_arcsec')
        display = ml_name if _ok(ml_name) else (ml_catalog if _ok(ml_catalog) else "Match")
        te_str = f" tE={ml_te:.1f}d" if ml_te and not pd.isna(ml_te) else ""
        sep_str = f" ({ml_sep:.1f}\")" if ml_sep and not pd.isna(ml_sep) else ""
        cards.append(_cell("Microlens", f"{display}{te_str}{sep_str}", hit=True))

    # ZTF cell
    ztf_type = payload.get('ztf_var_type')
    if _ok(ztf_type):
        ztf_p = payload.get('ztf_var_period')
        zp_str = f" P={ztf_p:.4f}d" if ztf_p and not pd.isna(ztf_p) else ""
        cards.append(_cell("ZTF", f"{ztf_type}{zp_str}", hit=True))

    # TNS cell
    tns_name = payload.get('tns_name')
    if _ok(tns_name):
        tns_type = payload.get('tns_type', '')
        cards.append(_cell("TNS", f"{tns_name} ({tns_type})" if tns_type else tns_name, hit=True))

    # ALeRCE cell
    alerce_cls = payload.get('alerce_lc_class')
    if _ok(alerce_cls):
        prob = payload.get('alerce_lc_prob')
        prob_str = f" ({prob:.0%})" if prob and not pd.isna(prob) else ""
        cards.append(_cell("ALeRCE", f"{alerce_cls}{prob_str}", hit=True))

    # X-ray cell
    xray = payload.get('xray_det')
    if xray:
        flux = payload.get('xray_flux')
        flux_str = f" {flux:.1e}" if flux and not pd.isna(flux) else ""
        catalogs = payload.get('xray_source_catalogs')
        catalog_str = f" ({catalogs})" if catalogs and str(catalogs).strip() else ""
        cards.append(_cell("X-ray", f"Detected{catalog_str}{flux_str}", hit=True))

    # Legacy nearest-SFR cell.  This is proximity metadata, not membership.
    sfr_name = payload.get('sfr_name')
    if _ok(sfr_name):
        sfr_sep = payload.get('sfr_sep_arcmin')
        sep_str = f" ({sfr_sep:.1f}')" if sfr_sep and not pd.isna(sfr_sep) else ""
        cards.append(_cell("Nearest SFR", f"{sfr_name}{sep_str}", hit=True))

    # Cluster cell
    cluster_name = payload.get('cluster_name')
    if _ok(cluster_name):
        cluster_dist = payload.get('cluster_dist_pc')
        d_str = f" ({cluster_dist:.0f} pc)" if cluster_dist and not pd.isna(cluster_dist) else ""
        cards.append(_cell("Cluster", f"{cluster_name}{d_str}", hit=True))

    # Separate cloud-environment overlap from stellar-association membership.
    environment_matches = payload.get('sfr_environment_matches')
    if _ok(environment_matches):
        cards.append(_cell("Cloud environment", str(environment_matches), hit=True))

    membership_class = payload.get('sfr_membership_class')
    membership_name = payload.get('sfr_membership_name')
    association_classes = {
        'catalog_confirmed_member',
        'kinematically_consistent_member',
        'dispersed_association_member',
    }
    if str(membership_class) in association_classes:
        label = str(membership_class).replace('_', ' ').title()
        name_str = f": {membership_name}" if _ok(membership_name) else ""
        cards.append(_cell("Association evidence", f"{label}{name_str}", hit=True))

    # BANYAN cells: mapped-SFR probability is association-specific; the legacy
    # global value remains explicitly labeled as global.
    mapped_sfr = payload.get('banyan_sfr_name')
    mapped_prob = payload.get('banyan_sfr_prob')
    if _ok(mapped_sfr):
        threshold = payload.get('sfr_membership_threshold', 0.90)
        mapped_hit = bool(
            mapped_prob is not None
            and not pd.isna(mapped_prob)
            and threshold is not None
            and not pd.isna(threshold)
            and float(mapped_prob) >= float(threshold)
        )
        prob_str = (
            f" ({mapped_prob:.0%})"
            if mapped_prob is not None and not pd.isna(mapped_prob)
            else ""
        )
        cards.append(
            _cell(
                "BANYAN mapped SFR",
                f"{mapped_sfr}{prob_str}",
                hit=mapped_hit,
            )
        )

    banyan_assoc = payload.get('banyan_best_assoc')
    banyan_fp = payload.get('banyan_field_prob')
    if _ok(banyan_assoc) and str(banyan_assoc).strip().lower() != 'field':
        fp_str = f" (P_field={banyan_fp:.0%})" if banyan_fp and not pd.isna(banyan_fp) else ""
        cards.append(_cell("BANYAN global", f"{banyan_assoc}{fp_str}", hit=True))

    # YSO class cell (skip generic classifications from IR color-color)
    yso_cls = payload.get('yso_class')
    if yso_cls and str(yso_cls).strip().lower() not in ('nan', '<na>', '', 'main sequence', 'unknown'):
        cards.append(_cell("YSO", str(yso_cls), hit=True))

    # Spectra cell
    has_spectrum = _coerce_bool(payload.get('has_spectrum'))
    if has_spectrum:
        sources = payload.get('spectrum_sources')
        src_str = f" ({sources})" if sources and not pd.isna(sources) else ""
        cards.append(_cell("Spectra", f"Available{src_str}", hit=True))

    # OGLE cell
    ogle_match = payload.get('period_ogle_match')
    if ogle_match:
        ogle_cls = payload.get('period_ogle_class', '')
        ogle_p = payload.get('period_ogle_days')
        ogle_sep = payload.get('period_ogle_sep_arcsec')
        p_str = f" P={ogle_p:.4f}d" if ogle_p and not pd.isna(ogle_p) else ""
        sep_str = f" ({ogle_sep:.1f}\")" if ogle_sep and not pd.isna(ogle_sep) else ""
        cards.append(_cell("OGLE", f"{ogle_cls}{p_str}{sep_str}" if ogle_cls else f"Match{p_str}{sep_str}", hit=True))

    # unWISE W1 variability cell
    w1_var = payload.get('unwise_w1_var')
    if w1_var:
        w1_z = payload.get('unwise_w1_zscore')
        z_str = f" (z={w1_z:.1f})" if w1_z and not pd.isna(w1_z) else ""
        cards.append(_cell("unWISE W1", f"Variable{z_str}", hit=True))

    # LTV trend cell
    ltv_slope = payload.get('ltv_slope')
    if ltv_slope is not None and not pd.isna(ltv_slope):
        ltv_diff = payload.get('ltv_max_diff')
        ltv_fap = payload.get('ltv_ls_fap')
        direction = "▲" if ltv_slope > 0 else "▼"
        diff_str = f" Δ{ltv_diff:.3f}mag" if ltv_diff and not pd.isna(ltv_diff) else ""
        fap_str = f" FAP={ltv_fap:.2e}" if ltv_fap and not pd.isna(ltv_fap) else ""
        cards.append(_cell("LTV", f"{direction}{ltv_slope:+.4f} mag/yr{diff_str}{fap_str}", hit=True))

    # IPHAS H-alpha excess cell
    ha_excess = payload.get('iphas_ha_excess')
    if ha_excess and not pd.isna(ha_excess) and float(ha_excess) > 0:
        cards.append(_cell("IPHAS Hα", f"excess={float(ha_excess):.2f}", hit=True))

    # Gaia epoch cell (non-hit, informational)
    epoch_n = payload.get('gaia_epoch_n_obs')
    if epoch_n and int(epoch_n) > 0:
        g_range = payload.get('gaia_epoch_g_range')
        r_str = f", dG={g_range:.2f}" if g_range and not pd.isna(g_range) else ""
        cards.append(_cell("Gaia epoch", f"{int(epoch_n)} obs{r_str}"))

    if not cards and not known:
        # No matches at all — emphasize "new"
        cards.append(html.Div([
            _value("No catalog matches found", hit=True),
        ], className='vetting-banner-cell'))

    # External links toolbar
    links = build_external_lookup_links(payload, radius_arcsec=radius_arcsec)
    links_row = None
    if links:
        link_els = []
        for label, url in links:
            link_els.append(html.A(
                label,
                href=url,
                target='_blank',
                rel='noopener noreferrer',
                className='vetting-banner-link',
            ))

        links_row = html.Div(link_els, className='vetting-banner-links')

    children = [
        html.Div(header_text, className=f'vetting-banner-header {banner_state}'),
        html.Div(cards, className='vetting-banner-grid with-links' if links_row else 'vetting-banner-grid'),
    ]
    if links_row:
        children.append(links_row)

    return html.Div(children, className=f'vetting-banner-shell {banner_state}')


def _keyboard_key(key_value: str | None) -> str:
    """Extract raw key token from encoded keyboard input value."""
    if not key_value:
        return ""
    return str(key_value).split("\t", 1)[0].strip()


def _format_large_integer_like_display(value) -> str:
    """Format integer-like values without scientific notation for display."""
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError):
        return s
    if d != d.to_integral_value():
        return s
    try:
        return format(d.to_integral_value(), "f")
    except Exception:
        return s


def _truncate_bottom_context_value(value: object, *, max_len: int = 84, tail_parts: int = 4) -> tuple[str, str]:
    """Return (full_text, display_text) for compact bottom-bar display."""
    full = str(value or "-").strip() or "-"
    if len(full) <= max_len:
        return full, full

    normalized = full.replace("\\", "/")
    if "/" in normalized:
        parts = [p for p in normalized.split("/") if p]
        prefix = "/" if normalized.startswith("/") else ""
        if len(parts) > tail_parts:
            display = f"{prefix}.../{'/'.join(parts[-tail_parts:])}"
            if len(display) <= max_len:
                return full, display

    head = max(12, max_len // 2 - 6)
    tail = max(12, max_len - head - 3)
    return full, f"{full[:head]}...{full[-tail:]}"


def _render_bottom_context(label: str, value: object) -> html.Div:
    """Render one labeled bottom-bar context item."""
    full, display = _truncate_bottom_context_value(value)
    return html.Div(
        [
            html.Span(f"{label}:", className='bottom-context-k'),
            html.Span(display, className='bottom-context-v', title=full),
        ],
        className='bottom-context-item',
    )
