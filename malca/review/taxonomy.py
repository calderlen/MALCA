"""Review taxonomy definitions and migration helpers."""

from __future__ import annotations

import json
import argparse
import sqlite3
from pathlib import Path
from typing import Any, Sequence


TAXONOMY_VERSION = 1

WORKFLOW_STATUS = ("unreviewed", "reviewed", "needs_followup")
DISPOSITIONS = (
    "keep",
    "reject",
    "ambiguous",
    "duplicate",
    "known_variable",
    "known_transient",
    "uncertain",
)
CLASSIFICATION_CONFIDENCE = (
    "morphology_only",
    "possible",
    "likely",
    "secure",
    "rejected",
    "ambiguous",
)

REVIEW_TAXONOMY_SQL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("workflow_status", "TEXT"),
    ("disposition", "TEXT"),
    ("morphology_primary", "TEXT"),
    ("morphology_secondary", "TEXT"),
    ("morphology_polarity", "TEXT"),
    ("morphology_recurrence", "TEXT"),
    ("baseline_behavior", "TEXT"),
    ("physical_primary", "TEXT"),
    ("physical_secondary", "TEXT"),
    ("classification_confidence", "TEXT"),
    ("priority_tags_json", "TEXT"),
    ("evidence_flags_json", "TEXT"),
    ("model_tags_json", "TEXT"),
    ("duplicate_of", "TEXT"),
    ("known_object_id", "TEXT"),
    ("known_object_source", "TEXT"),
    ("taxonomy_version", "INTEGER"),
    ("legacy_review_json", "TEXT"),
)

REVIEW_TAXONOMY_FIELDS: tuple[str, ...] = tuple(col for col, _dtype in REVIEW_TAXONOMY_SQL_COLUMNS)


def _entry(value: str, key: str, label: str) -> dict[str, str]:
    return {"value": value, "key": key, "label": label}


MORPHOLOGY_PRIMARY: tuple[dict[str, str], ...] = (
    _entry("artifact_or_bad_photometry", "x", "artifact / bad photometry"),
    _entry("nonvariable_or_low_snr", "n", "nonvariable / low SNR"),
    _entry("dimming_event", "d", "dimming event"),
    _entry("brightening_event", "b", "brightening event"),
    _entry("mixed_dip_and_burst", "m", "mixed dip + burst"),
    _entry("periodic", "p", "periodic"),
    _entry("quasi_periodic", "q", "quasi-periodic"),
    _entry("stochastic", "s", "stochastic"),
    _entry("long_term_trend", "l", "long-term trend"),
    _entry("complex_or_composite", "c", "complex / composite"),
    _entry("unclear", "u", "unclear"),
)

MORPHOLOGY_SECONDARY: dict[str, tuple[str, ...]] = {
    "artifact_or_bad_photometry": (
        "bad_photometry",
        "image_subtraction_artifact",
        "bad_reference_image",
        "saturation_or_bleed_trail",
        "diffraction_spike",
        "camera_specific_offset",
        "zero_point_failure",
        "isolated_bad_camera",
        "moving_object",
        "nearby_variable_contamination",
        "unresolved_blend",
        "crowded_field_blend",
        "alias_or_window_function",
        "insufficient_data",
    ),
    "nonvariable_or_low_snr": (
        "nonvariable",
        "low_snr",
        "insufficient_data",
        "poor_sampling",
        "marginal_variability",
    ),
    "dimming_event": (
        "single_dip",
        "big_dipper",
        "sharp_dip",
        "broad_dip",
        "boxy_dip",
        "asymmetric_dip",
        "symmetric_dip",
        "recurrent_dips",
        "periodic_dips",
        "quasi_periodic_dips",
        "aperiodic_dips",
        "stochastic_dips",
        "long_duration_low_state",
        "secular_dimming",
        "monotonic_dimming",
        "step_like_dimming",
        "dimming_with_recovery",
        "dimming_without_recovery",
        "multi_depth_dips",
        "color_dependent_dip",
        "possible_eclipse",
        "possible_occultation",
    ),
    "brightening_event": (
        "single_brightening",
        "single_symmetric_brightening",
        "single_asymmetric_brightening",
        "single_broad_brightening",
        "single_short_brightening",
        "multi_peak_brightening",
        "recurrent_brightenings",
        "recurrent_bursts",
        "stochastic_bursts",
        "quasi_periodic_bursts",
        "fast_rise_exponential_decay",
        "fast_rise_slow_decline",
        "slow_rise_slow_decline",
        "long_duration_outburst",
        "secular_brightening",
        "monotonic_brightening",
        "step_like_brightening",
        "brightening_with_recovery",
        "brightening_without_recovery",
        "brightening_on_variable_baseline",
        "possible_microlensing_event",
        "possible_flare",
        "possible_outburst",
    ),
    "mixed_dip_and_burst": (
        "dipper_plus_burster",
        "alternating_dips_and_bursts",
        "stochastic_dips_and_bursts",
        "periodic_plus_stochastic_events",
        "accretion_like_mixed_variability",
        "contaminated_mixed_variability",
    ),
    "periodic": (
        "eclipsing_like",
        "pulsator_like",
        "rotator_like",
        "ellipsoidal_like",
        "sinusoidal",
        "non_sinusoidal",
        "sawtooth",
        "multi_periodic",
        "double_wave",
        "contact_binary_like",
        "detached_binary_like",
        "semi_detached_binary_like",
        "heartbeat_like",
        "reflection_effect_like",
    ),
    "quasi_periodic": (
        "quasi_periodic_dimming",
        "quasi_periodic_brightening",
        "quasi_periodic_symmetric_variability",
        "quasi_periodic_long_cycle",
        "quasi_periodic_spot_modulation",
        "quasi_periodic_accretion_variability",
    ),
    "stochastic": (
        "stochastic_symmetric",
        "stochastic_dipper",
        "stochastic_burster",
        "stochastic_mixed",
        "stochastic_low_amplitude",
        "stochastic_high_amplitude",
        "flickering",
        "red_noise_like",
        "agn_like_stochastic",
        "yso_like_stochastic",
        "cv_like_stochastic",
    ),
    "long_term_trend": (
        "secular_dimming",
        "secular_brightening",
        "monotonic_trend",
        "long_duration_low_state",
        "long_duration_high_state",
        "state_change",
        "long_cycle",
        "quasi_periodic_long_cycle",
        "irregular_long_term_variability",
        "gradual_recovery",
        "no_recovery",
        "baseline_shift",
    ),
    "complex_or_composite": (
        "periodic_plus_dips",
        "periodic_plus_bursts",
        "trend_plus_events",
        "stochastic_plus_events",
        "multiple_event_types",
        "multi_state_variable",
        "blended_or_contaminated",
        "ambiguous_morphology",
        "insufficient_context",
    ),
    "unclear": ("unclear", "ambiguous_morphology", "insufficient_context"),
}

PHYSICAL_PRIMARY: tuple[dict[str, str], ...] = (
    _entry("young_stellar_object_or_pms", "y", "YSO / PMS"),
    _entry("massive_star_emission_line_or_mass_loss", "m", "massive / emission-line"),
    _entry("dust_obscuration_or_fading_variable", "d", "dust obscuration"),
    _entry("pulsating_variable", "p", "pulsating"),
    _entry("rotating_spotted_or_magnetic_variable", "r", "rotating / magnetic"),
    _entry("eclipsing_or_geometric_binary", "e", "eclipsing / geometric binary"),
    _entry("cataclysmic_or_compact_accretor", "c", "CV / compact accretor"),
    _entry("xray_or_high_energy_binary", "x", "X-ray / high-energy"),
    _entry("microlensing", "g", "microlensing"),
    _entry("flare_star_or_magnetically_active_star", "f", "flare / active star"),
    _entry("extragalactic_or_nuclear_variable", "a", "extragalactic / nuclear"),
    _entry("solar_system_or_moving_object", "o", "solar system / moving"),
    _entry("false_positive_or_contaminant", "z", "false positive / contaminant"),
    _entry("unknown", "u", "unknown"),
)

PHYSICAL_SECONDARY: dict[str, tuple[str, ...]] = {
    "young_stellar_object_or_pms": (
        "generic_yso_candidate",
        "pre_main_sequence_variable",
        "classical_t_tauri",
        "weak_line_t_tauri",
        "orion_variable",
        "herbig_ae_be_star",
        "pms_spotted_rotator",
        "pms_eclipsing_or_occulting_system",
        "yso_dipper",
        "aa_tau_like_dipper",
        "ux_orionis_like_dipper",
        "aperiodic_yso_dipper",
        "quasi_periodic_yso_dipper",
        "periodic_yso_dipper",
        "circumstellar_dust_dipper",
        "inner_disk_warp_occultation",
        "disk_occultation_candidate",
        "yso_burster",
        "accretion_burster",
        "stochastic_accretor",
        "fu_orionis_like",
        "ex_lupi_like",
        "yso_long_timescale_variable",
        "yso_mixed_dipper_burster",
    ),
    "eclipsing_or_geometric_binary": (
        "generic_eclipsing_binary",
        "algol_ea",
        "beta_lyrae_eb",
        "w_uma_ew",
        "detached_eclipsing_binary",
        "semi_detached_eclipsing_binary",
        "contact_binary",
        "overcontact_binary",
        "ellipsoidal_variable",
        "reflection_effect_binary",
        "heartbeat_star",
        "single_dip_eclipsing_binary_candidate",
        "multi_dip_eclipsing_binary_candidate",
        "long_period_eclipsing_binary",
        "disk_eclipse_system",
        "occulting_disk_binary",
        "transiting_exoplanet_candidate",
        "ambiguous_eclipsing_or_occulting_system",
    ),
    "microlensing": (
        "generic_microlensing_candidate",
        "pspl_microlensing_candidate",
        "blended_pspl_microlensing_candidate",
        "binary_lens_candidate",
        "caustic_crossing_candidate",
        "long_timescale_microlensing_candidate",
        "parallax_microlensing_candidate",
        "anomalous_microlensing_candidate",
        "possible_self_lensing_binary",
        "rejected_microlensing_candidate",
    ),
    "false_positive_or_contaminant": (
        "bad_photometry",
        "image_subtraction_artifact",
        "bad_reference_image",
        "saturation_or_bleed_trail",
        "diffraction_spike",
        "camera_specific_offset",
        "zero_point_failure",
        "isolated_bad_camera",
        "nearby_variable_contamination",
        "unresolved_blend",
        "crowded_field_blend",
        "alias_or_window_function",
        "seasonal_systematic",
        "processing_failure",
        "human_review_reject",
    ),
}

PRIORITY_TAGS = (
    "priority_dipper",
    "priority_big_dipper",
    "priority_recurrent_dipper",
    "priority_periodic_dipper",
    "priority_quasi_periodic_dipper",
    "priority_aperiodic_dipper",
    "priority_microlensing",
    "priority_anomalous_microlensing",
    "priority_stochastic_burster",
    "priority_yso_burster",
    "priority_long_term_variable",
    "priority_extreme_dimming",
    "priority_followup",
    "priority_spectrum",
    "priority_image_review",
    "priority_catalog_check",
    "priority_known_variable_comparison",
)

CANONICAL_ALIASES = {
    "herbig_ae_be_star": "herbig_ae_be_star",
    "ro_ap": "ro_ap",
    "moving_object": "moving_object",
    "secular_dimming": "secular_dimming",
    "secular_brightening": "secular_brightening",
}

SECONDARY_KEY_SEQUENCE = "123456789abcdefghijklmnopqrstuvwxyz"


def label_for(value: str | None) -> str:
    if not value:
        return ""
    return str(value).replace("_", " ")


def keyboard_payload() -> dict[str, Any]:
    return {
        "version": TAXONOMY_VERSION,
        "morphology_primary": list(MORPHOLOGY_PRIMARY),
        "morphology_secondary": {
            primary: [
                {"value": value, "key": SECONDARY_KEY_SEQUENCE[idx], "label": label_for(value)}
                for idx, value in enumerate(values)
                if idx < len(SECONDARY_KEY_SEQUENCE)
            ]
            for primary, values in MORPHOLOGY_SECONDARY.items()
        },
        "physical_primary": list(PHYSICAL_PRIMARY),
        "physical_secondary": {
            family: [
                {"value": value, "key": SECONDARY_KEY_SEQUENCE[idx], "label": label_for(value)}
                for idx, value in enumerate(values)
                if idx < len(SECONDARY_KEY_SEQUENCE)
            ]
            for family, values in PHYSICAL_SECONDARY.items()
        },
    }


def empty_taxonomy_selection() -> dict[str, Any]:
    return {
        "morphology_primary": None,
        "morphology_secondary": None,
        "morphology_polarity": None,
        "morphology_recurrence": None,
        "baseline_behavior": None,
        "physical_primary": None,
        "physical_secondary": None,
        "classification_confidence": None,
        "priority_tags": [],
        "evidence_flags": [],
        "model_tags": [],
        "disposition": None,
        "duplicate_of": None,
        "known_object_id": None,
        "known_object_source": None,
        "taxonomy_version": TAXONOMY_VERSION,
    }


def coerce_json_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            parsed = [value]
    else:
        parsed = value
    if not isinstance(parsed, (list, tuple, set)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def json_list(value: Any) -> str:
    return json.dumps(coerce_json_list(value), sort_keys=True, separators=(",", ":"))


def coerce_optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def normalize_selection(selection: dict[str, Any] | None) -> dict[str, Any]:
    out = empty_taxonomy_selection()
    if isinstance(selection, dict):
        out.update(selection)
    for key in (
        "morphology_primary",
        "morphology_secondary",
        "morphology_polarity",
        "morphology_recurrence",
        "baseline_behavior",
        "physical_primary",
        "physical_secondary",
        "classification_confidence",
        "disposition",
        "duplicate_of",
        "known_object_id",
        "known_object_source",
    ):
        out[key] = coerce_optional_text(out.get(key))
    for key in ("priority_tags", "evidence_flags", "model_tags"):
        out[key] = coerce_json_list(out.get(key))
    out["taxonomy_version"] = TAXONOMY_VERSION
    return out


def selection_from_review(review: dict[str, Any]) -> dict[str, Any]:
    return normalize_selection(
        {
            "morphology_primary": review.get("morphology_primary"),
            "morphology_secondary": review.get("morphology_secondary"),
            "morphology_polarity": review.get("morphology_polarity"),
            "morphology_recurrence": review.get("morphology_recurrence"),
            "baseline_behavior": review.get("baseline_behavior"),
            "physical_primary": review.get("physical_primary"),
            "physical_secondary": review.get("physical_secondary"),
            "classification_confidence": review.get("classification_confidence"),
            "priority_tags": review.get("priority_tags_json") or review.get("priority_tags"),
            "evidence_flags": review.get("evidence_flags_json") or review.get("evidence_flags"),
            "model_tags": review.get("model_tags_json") or review.get("model_tags"),
            "disposition": review.get("disposition"),
            "duplicate_of": review.get("duplicate_of"),
            "known_object_id": review.get("known_object_id"),
            "known_object_source": review.get("known_object_source"),
        }
    )


def derive_event_class(selection: dict[str, Any] | None) -> str:
    sel = normalize_selection(selection)
    physical = sel.get("physical_primary")
    primary = sel.get("morphology_primary")
    if physical == "microlensing":
        return "microlensing"
    if physical == "flare_star_or_magnetically_active_star":
        return "flare"
    if physical == "false_positive_or_contaminant" or primary == "artifact_or_bad_photometry":
        return "instrumental"
    if primary == "dimming_event":
        return "dipper"
    if primary == "long_term_trend":
        return "ltv"
    if primary:
        return primary
    return "unclassified"


def legacy_review_to_taxonomy(row: dict[str, Any]) -> dict[str, Any]:
    event_class = str(row.get("event_class") or "").strip()
    old_status = str(row.get("status") or "").strip()
    workflow_status = "needs_followup" if old_status == "needs_followup" else (
        "reviewed" if old_status and old_status != "unreviewed" else "unreviewed"
    )
    disposition = "keep" if workflow_status in {"reviewed", "needs_followup"} else None
    selection = empty_taxonomy_selection()
    selection["disposition"] = disposition

    if event_class == "dipper":
        selection["morphology_primary"] = "dimming_event"
        selection["priority_tags"] = ["priority_dipper"]
    elif event_class == "microlensing":
        selection["morphology_primary"] = "brightening_event"
        selection["physical_primary"] = "microlensing"
    elif event_class == "flare":
        selection["morphology_primary"] = "brightening_event"
        selection["physical_primary"] = "flare_star_or_magnetically_active_star"
    elif event_class == "ltv":
        selection["morphology_primary"] = "long_term_trend"
    elif event_class == "instrumental":
        selection["morphology_primary"] = "artifact_or_bad_photometry"
        selection["physical_primary"] = "false_positive_or_contaminant"

    normalized = normalize_selection(selection)
    normalized["workflow_status"] = workflow_status
    normalized["legacy_review_json"] = json.dumps(row, sort_keys=True, default=str)
    return normalized


def migrate_legacy_review_db(
    old_db: str | Path,
    new_db: str | Path,
    *,
    replace: bool = False,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Convert a legacy review DB into the taxonomy review schema."""
    import pandas as pd

    from malca.review.store import db_connect, upsert_candidates_frame
    from malca.review.sync import export_review_bundle

    old_path = Path(old_db).expanduser().resolve()
    new_path = Path(new_db).expanduser().resolve()
    if not old_path.exists():
        raise FileNotFoundError(f"Legacy DB not found: {old_path}")
    if old_path == new_path:
        raise ValueError("--input-review-db and --output-review-db must be different paths")
    if new_path.exists() and not replace:
        raise FileExistsError(f"Target DB exists; pass --replace to overwrite: {new_path}")
    if replace:
        for path in (new_path, Path(f"{new_path}-wal"), Path(f"{new_path}-shm")):
            if path.exists():
                path.unlink()

    with sqlite3.connect(old_path) as src_conn:
        candidates = pd.read_sql_query("SELECT * FROM candidates", src_conn)
        try:
            reviews = pd.read_sql_query("SELECT * FROM reviews", src_conn)
        except Exception:
            reviews = pd.DataFrame()
        try:
            review_history = pd.read_sql_query(
                "SELECT candidate_id, event_type, payload_json, reviewer, created_at FROM review_history",
                src_conn,
            )
        except Exception:
            review_history = pd.DataFrame()
        try:
            app_state = pd.read_sql_query("SELECT key, value, updated_at FROM app_state", src_conn)
        except Exception:
            app_state = pd.DataFrame()

    migrated_reviews = 0
    with db_connect(new_path) as dst_conn:
        if not candidates.empty:
            upsert_candidates_frame(dst_conn, candidates)

        for _, row in reviews.iterrows():
            data = {str(key): (None if pd.isna(value) else value) for key, value in row.to_dict().items()}
            candidate_id = str(data.get("candidate_id") or "").strip()
            if not candidate_id:
                continue
            mapped = legacy_review_to_taxonomy(data)
            interest_score = data.get("interest_score")
            try:
                interest_score = int(interest_score) if interest_score not in (None, "") else None
            except Exception:
                interest_score = None
            review_pass = data.get("review_pass")
            try:
                review_pass = max(1, int(review_pass)) if review_pass not in (None, "") else 1
            except Exception:
                review_pass = 1
            updated_at = str(data.get("updated_at") or "")
            if not updated_at:
                from malca.review.store import _utc_now

                updated_at = _utc_now()
            event_class = str(data.get("event_class") or "unclassified")
            workflow_status = str(mapped.get("workflow_status") or "unreviewed")
            insert_cols = [
                "candidate_id",
                "interest_score",
                "event_class",
                "review_pass",
                "notes",
                "status",
                "reviewer",
                *REVIEW_TAXONOMY_FIELDS,
                "updated_at",
            ]
            taxonomy_values = {
                "workflow_status": workflow_status,
                "disposition": mapped.get("disposition"),
                "morphology_primary": mapped.get("morphology_primary"),
                "morphology_secondary": mapped.get("morphology_secondary"),
                "morphology_polarity": mapped.get("morphology_polarity"),
                "morphology_recurrence": mapped.get("morphology_recurrence"),
                "baseline_behavior": mapped.get("baseline_behavior"),
                "physical_primary": mapped.get("physical_primary"),
                "physical_secondary": mapped.get("physical_secondary"),
                "classification_confidence": mapped.get("classification_confidence"),
                "priority_tags_json": json_list(mapped.get("priority_tags")),
                "evidence_flags_json": json_list(mapped.get("evidence_flags")),
                "model_tags_json": json_list(mapped.get("model_tags")),
                "duplicate_of": mapped.get("duplicate_of"),
                "known_object_id": mapped.get("known_object_id"),
                "known_object_source": mapped.get("known_object_source"),
                "taxonomy_version": TAXONOMY_VERSION,
                "legacy_review_json": mapped.get("legacy_review_json") or "{}",
            }
            placeholders = ", ".join(["?"] * len(insert_cols))
            dst_conn.execute(
                f"INSERT INTO reviews ({', '.join(insert_cols)}) VALUES ({placeholders})",
                (
                    candidate_id,
                    interest_score,
                    event_class,
                    review_pass,
                    "" if data.get("notes") is None else str(data.get("notes")),
                    workflow_status,
                    "" if data.get("reviewer") is None else str(data.get("reviewer")),
                    *(taxonomy_values[col] for col in REVIEW_TAXONOMY_FIELDS),
                    updated_at,
                ),
            )
            migrated_reviews += 1

        if not review_history.empty:
            review_history.to_sql("review_history", dst_conn, if_exists="append", index=False)
        if not app_state.empty:
            app_state.to_sql("app_state", dst_conn, if_exists="append", index=False)
        dst_conn.commit()

    result: dict[str, Any] = {
        "old_db": str(old_path),
        "new_db": str(new_path),
        "candidates": int(len(candidates)),
        "reviews": migrated_reviews,
        "review_history": int(len(review_history)),
        "app_state": int(len(app_state)),
    }
    if out_dir is not None:
        result["export"] = export_review_bundle(new_path, out_dir)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="malca review-taxonomy",
        description="Convert a legacy review DB into the taxonomy schema.",
    )
    parser.add_argument("--input-review-db", required=True, type=Path, help="Legacy review DB path")
    parser.add_argument("--output-review-db", required=True, type=Path, help="New taxonomy review DB path")
    parser.add_argument("--replace", action="store_true", help="Overwrite --output-review-db if it exists")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional review Git bundle export directory")
    args = parser.parse_args(argv)

    result = migrate_legacy_review_db(
        args.input_review_db,
        args.output_review_db,
        replace=bool(args.replace),
        out_dir=args.output_dir,
    )
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
