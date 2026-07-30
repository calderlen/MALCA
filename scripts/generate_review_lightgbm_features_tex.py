#!/usr/bin/env python3
"""Generate LaTeX feature tables from saved review LightGBM model artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MODEL_DIR = Path(
    "output/runs/dat3-full-extended_2026-07-01-v4/results/"
    "dipper_feature_selection/stats_plus_periodicity_dip_jump"
)
EIGHT_CLASS_MODEL_DIR = Path(
    "output/runs/dat3-full-extended_2026-07-01-v4/results/"
    "eight_class_ml_separability/stats_plus_periodicity_dip_jump_context"
)
DEFAULT_OUTPUT = Path("docs/review_lightgbm_features.tex")
EIGHT_CLASS_IMPORTANCE_OUTPUT = Path("docs/review_lightgbm_features_eight_class_by_importance.tex")

LABELS: dict[str, str] = {
    "stats_jd_start": r"JD start (day)",
    "stats_jd_end": r"JD end (day)",
    "stats_time_span_days": r"Time span (days)",
    "stats_n_unique_nights": r"Unique nights",
    "stats_duty_cycle_fraction": r"Duty cycle",
    "stats_file_points_total": r"Points total",
    "stats_file_points_kept_after_filter": r"Points kept after filter",
    "stats_cadence_mean_dt_days": r"Cadence mean $\Delta t$ (days)",
    "stats_cadence_median_dt_days": r"Cadence median $\Delta t$ (days)",
    "stats_cadence_p05_dt_days": r"Cadence $P_{05}$ $\Delta t$ (days)",
    "stats_cadence_p95_dt_days": r"Cadence $P_{95}$ $\Delta t$ (days)",
    "stats_photometry_band_mode": r"Photometry bands used",
    "stats_photometry_g_points": r"$g$-band points",
    "stats_photometry_v_points": r"$V$-band points",
    "stats_photometry_v_minus_g_offset_mag": r"$V$--$g$ offset applied (mag)",
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
    "stats_clipped_std_mag_3sigma_about_median": r"Clipped std (3$\sigma$ about median) (mag)",
    "stats_n_outliers_removed_robust_3sigma": r"Outliers removed (robust 3$\sigma$)",
    "stats_error_and_snr_stats_error_mean": r"Photometric error mean (mag)",
    "stats_error_and_snr_stats_error_median": r"Photometric error median (mag)",
    "stats_error_and_snr_stats_error_p05": r"Photometric error $P_{05}$ (mag)",
    "stats_error_and_snr_stats_error_p95": r"Photometric error $P_{95}$ (mag)",
    "stats_error_and_snr_stats_snr_median": r"SNR median",
    "stats_error_and_snr_stats_snr_p05": r"SNR $P_{05}$",
    "stats_error_and_snr_stats_snr_p95": r"SNR $P_{95}$",
    "stats_variability_reduced_chi2_vs_constant": r"Reduced $\chi^2$ vs.\ constant",
    "stats_variability_von_neumann_ratio": r"Inverse von Neumann ratio $1/\eta$",
    "stats_variability_roms": r"RoMS",
    "stats_variability_lag1_autocorr": r"Lag-1 autocorrelation $\rho$",
    "stats_variability_stetson_I": r"Stetson $I$",
    "stats_variability_stetson_J": r"Stetson $J$",
    "stats_variability_stetson_K": r"Stetson $K$",
    "stats_variability_stetson_L": r"Stetson $L$",
    "stats_variability_stetson_J_time": r"Stetson $J_\mathrm{time}$",
    "stats_variability_stetson_L_time": r"Stetson $L_\mathrm{time}$",
    "stats_variability_flux_asymmetry_m": r"Flux asymmetry $M$",
    "stats_variability_quasi_periodicity_q": r"Phase-template quasi-periodicity $Q$",
    "stats_variability_quasi_periodicity_n_points": r"$Q$ point count",
    "stats_variability_quasi_periodicity_populated_bins": r"$Q$ populated bins",
    "stats_variability_quasi_periodicity_bin_coverage": r"$Q$ bin coverage",
    "stats_variability_quasi_periodicity_template_amplitude": r"$Q$ template amplitude (mag)",
    "stats_variability_quasi_periodicity_raw_scatter": r"$Q$ raw scatter (mag)",
    "stats_variability_quasi_periodicity_resid_scatter": r"$Q$ residual scatter (mag)",
    "stats_variability_quasi_periodicity_scatter_ratio": r"$Q$ residual/raw scatter",
    "stats_variability_periodic_feature_period_days": r"Quasi-periodicity feature period (days)",
    "stats_variability_periodic_feature_period_source": r"Quasi-periodicity period source",
    "stats_variability_string_length_resid_total": r"String length total (mag)",
    "stats_variability_string_length_resid_mean_step": r"String length mean step (mag)",
    "stats_variability_string_length_resid_n_steps": r"String length step count",
    "stats_variability_lomb_scargle_best_period_days": r"Lomb--Scargle best period (days)",
    "stats_variability_lomb_scargle_peak_power": r"Lomb--Scargle peak power",
    "stats_variability_lomb_scargle_fap": r"Lomb--Scargle false-alarm probability",
    "stats_lafler_kinman_t_time": r"Lafler--Kinman $T(t)$",
    "stats_lafler_kinman_t_phase": r"Lafler--Kinman $T(\phi|P)$",
    "stats_lafler_kinman_delta": r"Lafler--Kinman $\delta$",
    "stats_trend_slope_mag_per_day": r"$\mathrm{d}m/\mathrm{d}t$ (mag/day)",
    "stats_trend_slope_mag_per_year": r"$\mathrm{d}m/\mathrm{d}t$ (mag/year)",
    "stats_trend_r2": r"Trend $R^2$",
    "stats_gp_drw_sigma": r"GP--DRW $\sigma$ (mag)",
    "stats_gp_drw_tau": r"GP--DRW $\tau$ (days)",
    "stats_iar_phi": r"IAR $\phi$",
    "stats_sf_ml_amplitude": r"Structure-function ML amplitude (mag)",
    "stats_sf_ml_gamma": r"Structure-function ML $\gamma$",
    "stats_psi_cs": r"$\psi_{\mathrm{CS}}$",
    "stats_psi_eta": r"$\psi_{\eta}$",
    "stats_con": r"Con statistic",
    "stats_intrinsic_sigma_mag": r"Intrinsic $\sigma$ (mag)",
    "stats_amplitude": r"Amplitude (mag)",
    "stats_percent_amplitude": r"Percent amplitude",
    "stats_first_mag": r"First $m$ (mag)",
    "stats_max_slope": r"Max slope (mag/day)",
    "stats_median_abs_dev": r"Median absolute deviation (mag)",
    "stats_gskew": r"$g$-skew",
    "stats_meanvariance": r"Mean/variance",
    "stats_median_brp": r"Median BRP",
    "stats_constancy_p_value": r"Constancy $p$-value",
    "stats_q31": r"$Q_{31}$ (mag)",
    "stats_rcs": r"RCS",
    "stats_delta_mag_fid": r"$\Delta m_\mathrm{fid}$ (mag)",
    "stats_beyond_1_std": r"Beyond 1-$\sigma$ fraction",
    "stats_small_kurtosis": r"Small kurtosis",
    "stats_pair_slope_trend": r"Pair slope trend",
    "stats_ahl_ratio": r"AHL bright/faint ratio",
    "stats_skew": r"Skewness",
    "stats_anderson_darling": r"Anderson--Darling statistic",
    "stats_eb_rminima": r"EB minimum depth ratio",
    "stats_eb_primary_min_depth": r"EB primary minimum depth",
    "stats_eb_secondary_min_depth": r"EB secondary minimum depth",
    "stats_harmonics_mse": r"Harmonic model MSE (mag$^2$)",
    "stats_harmonics_order": r"Recommended harmonic order",
    "stats_harmonics_a0": r"Harmonic zero-point $A_0$ (mag)",
    "stats_harmonics_model_amplitude": r"Harmonic model amplitude (mag)",
    "stats_harmonics_reduced_chi2": r"Harmonic reduced $\chi^2$",
    "stats_mhps_high": r"MHPS high statistic",
    "stats_mhps_low": r"MHPS low statistic",
    "stats_mhps_ratio": r"MHPS ratio",
    "stats_camera_loo_corr_median": r"Leave-one-camera correlation median",
    "stats_camera_loo_corr_min": r"Leave-one-camera correlation minimum",
    "stats_camera_loo_rms_max": r"Leave-one-camera RMS maximum",
    "stats_window_alias_period_1": r"Window alias period 1 (days)",
    "stats_window_alias_period_2": r"Window alias period 2 (days)",
    "stats_window_alias_period_3": r"Window alias period 3 (days)",
    "stats_window_alias_period_4": r"Window alias period 4 (days)",
    "stats_window_alias_period_5": r"Window alias period 5 (days)",
    "stats_window_alias_power_1": r"Window alias power 1",
    "stats_window_alias_power_2": r"Window alias power 2",
    "stats_window_alias_power_3": r"Window alias power 3",
    "stats_window_alias_power_4": r"Window alias power 4",
    "stats_window_alias_power_5": r"Window alias power 5",
    "periodicity_period": r"Selected periodicity period (days)",
    "periodicity_method": r"Period-search method",
    "periodicity_harmonic_factor": r"Harmonic factor",
    "periodicity_harmonic_objective": r"Harmonic objective score",
    "periodicity_scatter_ratio": r"Phase-folded scatter ratio",
    "periodicity_alias_flag": r"Period alias flag",
    "periodicity_bootstrap_sig": r"Period bootstrap significance",
    "lsp_bootstrap_sig": r"Lomb--Scargle bootstrap significance",
    "lsp_is_alias": r"Lomb--Scargle alias flag",
    "pdm_theta": r"PDM $\theta$",
    "pdm_snr": r"PDM SNR",
    "pdm_bootstrap_sig": r"PDM bootstrap significance",
    "ce_entropy": r"Conditional entropy",
    "ce_snr": r"Conditional-entropy SNR",
    "ce_bootstrap_sig": r"Conditional-entropy bootstrap significance",
    "dip_best_morph": r"Best dip morphology class",
    "dip_best_delta_bic": r"Best dip $\Delta$BIC",
    "dip_best_width_param": r"Best dip width parameter",
    "dip_symmetry_score": r"Dip symmetry score",
    "dip_best_amp": r"Best dip amplitude (mag)",
    "dip_bayes_factor": r"Dip Bayes factor",
    "dip_best_p": r"Best dip significance $p$",
    "dip_max_event_prob": r"Maximum dip event probability",
    "dip_count": r"Dip event count",
    "dip_run_count": r"Dip run count",
    "dip_max_run_points": r"Longest dip run (points)",
    "dip_max_run_duration": r"Longest dip run duration (days)",
    "dip_max_run_sum": r"Longest dip run flux sum",
    "dip_max_run_max": r"Longest dip run peak depth",
    "dip_max_run_cameras": r"Longest dip run camera count",
    "dip_max_log_bf_local": r"Maximum local log Bayes factor",
    "dip_is_single_event": r"Single-dip flag",
    "dip_inter_event_spacing_median": r"Median dip inter-event spacing (days)",
    "dip_inter_event_spacing_std": r"Dip inter-event spacing std.\ (days)",
    "dip_amplitude_consistency": r"Dip amplitude consistency",
    "dip_duration_consistency": r"Dip duration consistency",
    "dipper_n_dips": r"Total dipper dips",
    "dipper_n_valid_dips": r"Valid dipper dips",
    "jump_best_morph": r"Best jump morphology class",
    "jump_best_delta_bic": r"Best jump $\Delta$BIC",
    "jump_best_width_param": r"Best jump width parameter",
    "jump_best_amp": r"Best jump amplitude (mag)",
    "jump_bayes_factor": r"Jump Bayes factor",
    "jump_best_p": r"Best jump significance $p$",
    "jump_max_event_prob": r"Maximum jump event probability",
    "jump_count": r"Jump event count",
    "jump_run_count": r"Jump run count",
    "jump_max_run_points": r"Longest jump run (points)",
    "jump_max_run_duration": r"Longest jump run duration (days)",
    "jump_max_run_sum": r"Longest jump run flux sum",
    "jump_max_run_max": r"Longest jump run peak amplitude",
    "jump_max_run_cameras": r"Longest jump run camera count",
    "jump_max_log_bf_local": r"Maximum local log Bayes factor",
    "jump_is_single_event": r"Single-jump flag",
    "jump_inter_event_spacing_median": r"Median jump inter-event spacing (days)",
    "jump_inter_event_spacing_std": r"Jump inter-event spacing std.\ (days)",
    "jump_amplitude_consistency": r"Jump amplitude consistency",
    "jump_duration_consistency": r"Jump duration consistency",
    "jumper_n_jumps": r"Total jumper events",
    "jumper_n_valid_jumps": r"Valid jumper events",
    "bprp0": r"Gaia BP--RP color",
    "derived_mrp": r"Derived absolute $M_\mathrm{RP}$",
    "ruwe": r"Gaia RUWE",
    "parallax_snr": r"Gaia parallax SNR",
    "derived_j_k": r"2MASS $J-K_\mathrm{s}$ color",
    "w1_w2": r"WISE $W1-W2$ color",
    "w1_w3": r"WISE $W1-W3$ color",
    "w2_w3": r"WISE $W2-W3$ color",
    "wise_w3_error": r"WISE $W3$ magnitude error",
    "wise_w4_error": r"WISE $W4$ magnitude error",
    "wise_w3_missing": r"Missing or invalid WISE $W3$ uncertainty flag",
    "wise_w4_missing": r"Missing or invalid WISE $W4$ uncertainty flag",
    "sed_alpha": r"SED power-law index $\alpha$",
    "tess_flux_range": r"TESS flux range",
}

for i in range(1, 8):
    LABELS[f"stats_harmonics_a{i}"] = rf"Harmonic cosine coefficient $a_{i}$ (mag)"
    LABELS[f"stats_harmonics_b{i}"] = rf"Harmonic sine coefficient $b_{i}$ (mag)"
    LABELS[f"stats_harmonics_mag_{i}"] = rf"Harmonic amplitude $A_{i}$ (mag)"
    if i >= 2:
        LABELS[f"stats_harmonics_phase_{i}"] = rf"Harmonic phase $\phi_{i}$ (rad)"
        LABELS[f"stats_harmonics_r{i}1"] = rf"Harmonic amplitude ratio $R_{i}1$"

CATEGORY_ORDER = [
    "Coverage \\& cadence",
    "Photometry",
    "Photometric error \\& SNR",
    "Variability",
    "Trend",
    "Harmonics",
    "Eclipsing-binary diagnostics",
    "Stochastic models",
    "MHPS",
    "Window aliases",
    "Camera diagnostics",
    "ALeRCE-style statistics",
    "Period search",
    "Dip detection",
    "Jump detection",
    "Astrophysical context",
]


def category(feature: str) -> str:
    if feature.startswith("stats_cadence_") or feature in {
        "stats_jd_start",
        "stats_jd_end",
        "stats_time_span_days",
        "stats_n_unique_nights",
        "stats_duty_cycle_fraction",
        "stats_file_points_total",
        "stats_file_points_kept_after_filter",
    }:
        return "Coverage \\& cadence"
    if feature.startswith("stats_photometry_") or feature.startswith("stats_clipped_") or feature == "stats_n_outliers_removed_robust_3sigma":
        return "Photometry"
    if feature.startswith("stats_error_and_snr_"):
        return "Photometric error \\& SNR"
    if feature.startswith("stats_variability_") or feature.startswith("stats_lafler_"):
        return "Variability"
    if feature.startswith("stats_trend_"):
        return "Trend"
    if feature.startswith("stats_harmonics_"):
        return "Harmonics"
    if feature.startswith("stats_eb_"):
        return "Eclipsing-binary diagnostics"
    if feature.startswith("stats_gp_") or feature.startswith("stats_iar_") or feature.startswith("stats_sf_") or feature in {"stats_psi_cs", "stats_psi_eta", "stats_con"}:
        return "Stochastic models"
    if feature.startswith("stats_mhps_"):
        return "MHPS"
    if feature.startswith("stats_window_alias_"):
        return "Window aliases"
    if feature.startswith("stats_camera_"):
        return "Camera diagnostics"
    if feature.startswith("stats_"):
        return "ALeRCE-style statistics"
    if feature.startswith("periodicity_") or feature.startswith("lsp_") or feature.startswith("pdm_") or feature.startswith("ce_"):
        return "Period search"
    if feature.startswith("dip") or feature.startswith("dipper_"):
        return "Dip detection"
    if feature.startswith("jump") or feature.startswith("jumper_"):
        return "Jump detection"
    return "Astrophysical context"


def latex_escape(name: str) -> str:
    return name.replace("_", r"\_").replace("%", r"\%")


def latex_feature_name(name: str) -> str:
    """Use breakable monospace so long feature names do not spill into the next column."""
    return r"\url{" + name + "}"


def short_category(feature: str) -> str:
    """Compact category labels for narrow table columns."""
    mapping = {
        "Coverage \\& cadence": "Coverage",
        "Photometry": "Photometry",
        "Photometric error \\& SNR": "Err/SNR",
        "Variability": "Variability",
        "Trend": "Trend",
        "Harmonics": "Harmonics",
        "Eclipsing-binary diagnostics": "EB",
        "Stochastic models": "Stochastic",
        "MHPS": "MHPS",
        "Window aliases": "Aliases",
        "Camera diagnostics": "Camera",
        "ALeRCE-style statistics": "ALeRCE",
        "Period search": "Period",
        "Dip detection": "Dip",
        "Jump detection": "Jump",
        "Astrophysical context": "Context",
    }
    return mapping.get(category(feature), category(feature))


def describe(feature: str) -> str:
    if feature in LABELS:
        return LABELS[feature]
    pretty = feature.replace("stats_", "").replace("_", " ")
    return pretty[0].upper() + pretty[1:]


def _load_importance(model_dir: Path) -> dict[str, dict[str, float]]:
    for name in ("feature_importance_gain_percent.csv", "feature_importance_gain.csv"):
        path = model_dir / name
        if not path.exists():
            continue
        rows: dict[str, dict[str, float]] = {}
        import csv

        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                feature = str(row["feature"])
                rows[feature] = {
                    "gain": float(row["gain"]),
                    "split": float(row.get("split") or 0.0),
                    "gain_percent": float(row["gain_percent"]) if row.get("gain_percent") else None,
                }
        if rows and all(value["gain_percent"] is None for value in rows.values()):
            total_gain = sum(value["gain"] for value in rows.values())
            for value in rows.values():
                value["gain_percent"] = 100.0 * value["gain"] / total_gain if total_gain else 0.0
        return rows
    raise FileNotFoundError(f"No feature-importance CSV found under {model_dir}")


def _sort_features(
    features: list[str],
    *,
    sort_by: str,
    model_dir: Path,
) -> list[str]:
    if sort_by == "category":
        category_rank = {name: idx for idx, name in enumerate(CATEGORY_ORDER)}
        return sorted(
            features,
            key=lambda feature: (category_rank.get(category(feature), 999), feature),
        )
    if sort_by == "importance":
        importance = _load_importance(model_dir)
        missing = [feature for feature in features if feature not in importance]
        if missing:
            raise KeyError(f"Feature-importance CSV is missing {len(missing)} model features: {missing[:5]}")
        return sorted(features, key=lambda feature: (-importance[feature]["gain"], feature))
    raise ValueError(f"Unsupported sort mode: {sort_by!r}")


def _caption_and_label(
    *,
    sort_by: str,
    target_column: str | None,
    label_classes: list[str] | None,
) -> tuple[str, str]:
    if target_column == "human_eight_class_label":
        classes = ", ".join(label_classes or [])
        target_tex = latex_escape(target_column)
        if sort_by == "importance":
            caption = (
                r"\caption{Input features for the eight-class MALCA review LightGBM classifier "
                + r"(\texttt{" + target_tex + r"}), ordered by LightGBM gain-based feature importance. "
                + "Classes: " + classes + ". "
                + r"Categorical features are encoded as integer category indices for LightGBM.} \\"
            )
            return caption, "tab:review-lightgbm-features-eight-class-importance"
        caption = (
            r"\caption{Input features for the eight-class MALCA review LightGBM classifier "
            + r"(\texttt{" + target_tex + r"}). Classes: " + classes + r".} \\"
        )
        return caption, "tab:review-lightgbm-features-eight-class"
    if sort_by == "importance":
        return (
            r"\caption{Input features for the MALCA review LightGBM classifier "
            r"(\texttt{stats\_plus\_periodicity\_dip\_jump}), ordered by LightGBM gain-based feature importance.} \\",
            "tab:review-lightgbm-features-importance",
        )
    return (
        r"\caption{Input features for the MALCA review LightGBM classifier "
        r"(\texttt{stats\_plus\_periodicity\_dip\_jump} feature set). "
        r"Features are drawn from light-curve statistics (\texttt{stats\_*}), native periodicity/dip/jump measurements, "
        r"and external catalog context. Categorical features are encoded as integer category indices for LightGBM.} \\",
        "tab:review-lightgbm-features",
    )


def generate(
    model_dir: Path,
    output_path: Path,
    *,
    sort_by: str = "category",
    twocolumn_doc: bool = False,
    revtex: bool = False,
) -> None:
    metadata_path = model_dir / "metadata.json"
    meta = json.loads(metadata_path.read_text())
    features = list(meta["feature_columns"])
    categorical = set(meta.get("categorical_features", []))
    target_column = meta.get("target_column")
    label_classes = meta.get("label_classes")
    sorted_features = _sort_features(features, sort_by=sort_by, model_dir=model_dir)
    importance = _load_importance(model_dir) if sort_by == "importance" else {}

    caption, label = _caption_and_label(
        sort_by=sort_by,
        target_column=target_column,
        label_classes=label_classes,
    )

    preamble = [
        "% Auto-generated by scripts/generate_review_lightgbm_features_tex.py",
        f"% Source model dir: {model_dir.as_posix()}",
        f"% Target column: {target_column}",
        f"% Sort order: {sort_by}",
        f"% Total features: {len(features)}",
        "%",
        "% Required preamble packages:",
        "%   \\usepackage{array}",
        "%   \\usepackage{booktabs}",
        "%   \\usepackage{longtable}",
        "%   \\usepackage{xurl}   % breakable \\url{...} feature names",
        "%",
        "% Do not wrap this file in \\begin{table}...\\end{table}; use \\input directly.",
    ]
    twocolumn_cmd: str | None = None
    if twocolumn_doc:
        onecolumn_cmd = r"\onecolumngrid" if revtex else r"\onecolumn"
        twocolumn_cmd = r"\twocolumngrid" if revtex else r"\twocolumn"
        preamble.extend(
            [
                "% Two-column note: longtable page breaks call \\clearpage, which would restore",
                "% two-column mode mid-table and produce two side-by-side broken columns. The",
                "% \\clearpage redefinition below keeps full-width mode active for every page.",
                "%",
                "% revtex / APS documents: pass --revtex to this script (uses onecolumngrid).",
                "%",
                "% Remove this whole block if your document is single-column.",
                r"\clearpage",
                onecolumn_cmd,
                r"\begingroup",
                r"\let\malca@saved@clearpage\clearpage",
                rf"\def\clearpage{{\malca@saved@clearpage{onecolumn_cmd}}}",
                r"\setlength{\LTcapwidth}{\textwidth}",
            ]
        )
    preamble.extend(
        [
            r"\begingroup",
            r"\footnotesize",
            r"\urlstyle{tt}",
            r"\setlength{\tabcolsep}{3pt}",
            r"\setlength{\LTleft}{0pt}",
            r"\setlength{\LTright}{0pt}",
            r"\setlength{\emergencystretch}{1.5em}",
            r"\renewcommand{\arraystretch}{1.05}",
            "",
        ]
    )

    if sort_by == "importance":
        # All fixed-width p columns; fractions sum to 0.83\linewidth, leaving room for tabcolsep.
        colspec = (
            r"@{}"
            r">{\raggedleft\arraybackslash}p{0.035\linewidth}"
            r">{\raggedright\arraybackslash}p{0.22\linewidth}"
            r">{\raggedright\arraybackslash}p{0.31\linewidth}"
            r">{\raggedright\arraybackslash}p{0.10\linewidth}"
            r">{\raggedleft\arraybackslash}p{0.065\linewidth}"
            r">{\centering\arraybackslash}p{0.05\linewidth}"
            r"@{}"
        )
        header = (
            r"\textbf{Rank} & \textbf{Feature name} & \textbf{Description} & "
            r"\textbf{Category} & \textbf{Gain (\%)} & \textbf{Type} \\"
        )
        continued_cols = "6"
    else:
        colspec = (
            r"@{}"
            r">{\raggedright\arraybackslash}p{0.17\linewidth}"
            r">{\raggedright\arraybackslash}p{0.20\linewidth}"
            r">{\raggedright\arraybackslash}p{0.49\linewidth}"
            r">{\centering\arraybackslash}p{0.06\linewidth}"
            r"@{}"
        )
        header = r"\textbf{Category} & \textbf{Feature name} & \textbf{Description} & \textbf{Type} \\"
        continued_cols = "4"

    lines = preamble + [
        rf"\begin{{longtable}}{{{colspec}}}",
        caption,
        rf"\label{{{label}}} \\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        "",
        rf"\multicolumn{{{continued_cols}}}{{c}}{{\tablename\ \thetable\ -- continued from previous page}} \\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        "",
        r"\midrule",
        rf"\multicolumn{{{continued_cols}}}{{r}}{{Continued on next page}} \\",
        r"\midrule",
        r"\endfoot",
        "",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    current_cat: str | None = None
    for rank, feature in enumerate(sorted_features, start=1):
        cat = category(feature)
        feature_type = "Cat." if feature in categorical else "Num."
        if sort_by == "importance":
            gain_percent = importance[feature]["gain_percent"]
            lines.append(
                f"{rank} & {latex_feature_name(feature)} & {describe(feature)} & "
                f"{short_category(feature)} & {gain_percent:.2f} & {feature_type} \\\\"
            )
            continue

        cat_cell = cat if cat != current_cat else ""
        current_cat = cat
        lines.append(
            f"{cat_cell} & {latex_feature_name(feature)} & {describe(feature)} & {feature_type} \\\\"
        )

    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")
    if twocolumn_doc:
        assert twocolumn_cmd is not None
        lines.extend([r"\endgroup", "", r"\clearpage", twocolumn_cmd, ""])
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing metadata.json and feature-importance CSVs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output .tex path",
    )
    parser.add_argument(
        "--sort-by",
        choices=("category", "importance"),
        default="category",
        help="Feature ordering in the table",
    )
    parser.add_argument(
        "--eight-class",
        action="store_true",
        help="Shortcut for the eight-class model sorted by importance",
    )
    parser.add_argument(
        "--revtex",
        action="store_true",
        help="Use \\onecolumngrid/\\twocolumngrid instead of \\onecolumn/\\twocolumn",
    )
    parser.add_argument(
        "--single-column-doc",
        action="store_true",
        help="Do not emit \\onecolumn/\\twocolumn wrappers",
    )
    args = parser.parse_args()

    model_dir = EIGHT_CLASS_MODEL_DIR if args.eight_class else args.model_dir
    output_path = EIGHT_CLASS_IMPORTANCE_OUTPUT if args.eight_class else args.output
    sort_by = "importance" if args.eight_class else args.sort_by
    twocolumn_doc = args.eight_class and not args.single_column_doc

    generate(
        model_dir,
        output_path,
        sort_by=sort_by,
        twocolumn_doc=twocolumn_doc,
        revtex=args.revtex,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
