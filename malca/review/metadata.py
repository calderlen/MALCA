from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import quote, quote_plus

import pandas as pd


# Grouped metadata fields: list of (group_name, fields) where each field is
# a (label, key) tuple.  Groups are rendered as collapsible sections in the
# Dash app and as headed blocks in the TUI.
REVIEW_METADATA_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Identification", [
        ("ASAS-SN ID", "asas_sn_id"),
        ("Path", "path"),
        ("Gaia ID", "gaia_id"),
        ("RA", "ra"),
        ("Dec", "dec"),
        ("2MASS ID", "tmass_id"),
        ("AllWISE ID", "allwise_id"),
    ]),
    ("Crossmatch", [
        ("VSX class", "vsx_class"),
        ("VSX sep (arcsec)", "vsx_sep_arcsec"),
        ("catalog_match", "catalog_match"),
        ("SFR name", "sfr_name"),
        ("SFR sep (arcmin)", "sfr_sep_arcmin"),
        ("Near SFR", "near_sfr"),
        ("Cluster name", "cluster_name"),
        ("Cluster dist (pc)", "cluster_dist_pc"),
        ("Cluster age (Myr)", "cluster_age_myr"),
    ]),
    ("Vetting", [
        ("Known?", "vetting_likely_known"),
        ("SIMBAD ID", "simbad_main_id"),
        ("SIMBAD type", "simbad_otype"),
        ("SIMBAD refs", "simbad_nbref"),
        ("SIMBAD sep (\")", "simbad_sep_arcsec"),
        ("Gaia variable", "gaia_var_flag"),
        ("Gaia var class", "gaia_var_class"),
        ("Gaia var score", "gaia_var_score"),
        ("Gaia EB period (d)", "gaia_eb_period"),
        ("Gaia EB morph", "gaia_eb_morph"),
        ("Gaia EB ranking", "gaia_eb_global_ranking"),
        ("Gaia epoch obs", "gaia_epoch_n_obs"),
        ("Gaia G range", "gaia_epoch_g_range"),
        ("ASAS-SN var name", "asassn_var_name"),
        ("ASAS-SN var type", "asassn_var_type"),
        ("ASAS-SN period", "asassn_var_period"),
        ("ZTF var type", "ztf_var_type"),
        ("ZTF var period (d)", "ztf_var_period"),
        ("ZTF var amp", "ztf_var_amp"),
        ("TNS name", "tns_name"),
        ("TNS type", "tns_type"),
        ("TNS redshift", "tns_redshift"),
        ("TNS disc date", "tns_disc_date"),
        ("ALeRCE OID", "alerce_oid"),
        ("ALeRCE ndet", "alerce_ndet"),
        ("ALeRCE LC class", "alerce_lc_class"),
        ("ALeRCE LC prob", "alerce_lc_prob"),
        ("ALeRCE stamp class", "alerce_stamp_class"),
        ("ALeRCE stamp prob", "alerce_stamp_prob"),
        ("Gaia epoch avail", "gaia_epoch_available"),
        ("X-ray detected", "xray_det"),
        ("X-ray flux", "xray_flux"),
        ("X-ray sep (\")", "xray_sep_arcsec"),
        ("PM cluster offset", "pm_cluster_offset_sigma"),
    ]),
    ("Light Curve Basics", [
        ("n_points", "n_points"),
        ("JD first", "jd_first"),
        ("JD last", "jd_last"),
        ("Cadence median (days)", "cadence_median_days"),
        ("n_cameras", "n_cameras"),
        ("Camera IDs", "camera_ids"),
        ("Camera min points", "camera_min_points"),
        ("Camera max points", "camera_max_points"),
        ("Baseline mag", "baseline_mag"),
        ("Baseline source", "baseline_source"),
        ("Trigger mode", "trigger_mode"),
    ]),
    ("Periodicity", [
        ("periodic_flag", "periodic_flag"),
        ("periodicity_score", "periodicity_score"),
        ("lsp_power", "lsp_power"),
        ("lsp_period", "lsp_period"),
        ("lsp_bootstrap_sig", "lsp_bootstrap_sig"),
        ("lsp_is_alias", "lsp_is_alias"),
        ("lsp_is_significant", "lsp_is_significant"),
    ]),
    ("Dip Detection", [
        ("dip_significant", "dip_significant"),
        ("dip_best_morph", "dip_best_morph"),
        ("dip_best_log_bf", "dip_best_log_bf"),
        ("dip_best_delta_bic", "dip_best_delta_bic"),
        ("dip_best_width_param", "dip_best_width_param"),
        ("dip_symmetry_score", "dip_symmetry_score"),
        ("dip_best_amp", "dip_best_amp"),
        ("dip_best_t0", "dip_best_t0"),
        ("dip_best_alpha", "dip_best_alpha"),
        ("dip_best_tau", "dip_best_tau"),
        ("dip_bayes_factor", "dip_bayes_factor"),
        ("dip_best_p", "dip_best_p"),
        ("dip_best_mag_event", "dip_best_mag_event"),
        ("dip_trigger_max", "dip_trigger_max"),
        ("dip_max_event_prob", "dip_max_event_prob"),
        ("dip_trigger_threshold", "dip_trigger_threshold"),
    ]),
    ("Dip Runs", [
        ("dip_count", "dip_count"),
        ("dip_run_count", "dip_run_count"),
        ("dip_max_run_points", "dip_max_run_points"),
        ("dip_max_run_duration", "dip_max_run_duration"),
        ("dip_max_run_sum", "dip_max_run_sum"),
        ("dip_max_run_max", "dip_max_run_max"),
        ("dip_max_run_cameras", "dip_max_run_cameras"),
        ("dip_max_log_bf_local", "dip_max_log_bf_local"),
    ]),
    ("Jump Detection", [
        ("jump_significant", "jump_significant"),
        ("jump_best_morph", "jump_best_morph"),
        ("jump_best_log_bf", "jump_best_log_bf"),
        ("jump_best_delta_bic", "jump_best_delta_bic"),
        ("jump_best_width_param", "jump_best_width_param"),
        ("jump_best_amp", "jump_best_amp"),
        ("jump_best_t0", "jump_best_t0"),
        ("jump_best_alpha", "jump_best_alpha"),
        ("jump_best_tau", "jump_best_tau"),
        ("jump_bayes_factor", "jump_bayes_factor"),
        ("jump_best_p", "jump_best_p"),
        ("jump_best_mag_event", "jump_best_mag_event"),
        ("jump_trigger_max", "jump_trigger_max"),
        ("jump_max_event_prob", "jump_max_event_prob"),
        ("jump_trigger_threshold", "jump_trigger_threshold"),
    ]),
    ("Jump Runs", [
        ("jump_count", "jump_count"),
        ("jump_run_count", "jump_run_count"),
        ("jump_max_run_points", "jump_max_run_points"),
        ("jump_max_run_duration", "jump_max_run_duration"),
        ("jump_max_run_sum", "jump_max_run_sum"),
        ("jump_max_run_max", "jump_max_run_max"),
        ("jump_max_run_cameras", "jump_max_run_cameras"),
        ("jump_max_log_bf_local", "jump_max_log_bf_local"),
    ]),
    ("Dip Recurrence", [
        ("dip_is_single_event", "dip_is_single_event"),
        ("dip_inter_event_spacing_median", "dip_inter_event_spacing_median"),
        ("dip_inter_event_spacing_std", "dip_inter_event_spacing_std"),
        ("dip_amplitude_consistency", "dip_amplitude_consistency"),
        ("dip_duration_consistency", "dip_duration_consistency"),
    ]),
    ("Jump Recurrence", [
        ("jump_is_single_event", "jump_is_single_event"),
        ("jump_inter_event_spacing_median", "jump_inter_event_spacing_median"),
        ("jump_inter_event_spacing_std", "jump_inter_event_spacing_std"),
        ("jump_amplitude_consistency", "jump_amplitude_consistency"),
        ("jump_duration_consistency", "jump_duration_consistency"),
    ]),
    ("Event Scoring", [
        ("dipper_score", "dipper_score"),
        ("dipper_n_dips", "dipper_n_dips"),
        ("dipper_n_valid_dips", "dipper_n_valid_dips"),
        ("jumper_score", "jumper_score"),
        ("jumper_n_jumps", "jumper_n_jumps"),
        ("jumper_n_valid_jumps", "jumper_n_valid_jumps"),
    ]),
    ("Stellar Parameters", [
        ("G mag", "phot_g_mean_mag"),
        ("BP-RP", "bp_rp"),
        ("BP-RP excess", "bp_rp_excess_corr"),
        ("ruwe", "ruwe"),
        ("RV (km/s)", "radial_velocity"),
        ("RV amp robust (km/s)", "rv_amplitude_robust"),
        ("high_ruwe_flag", "high_ruwe_flag"),
        ("Fidelity", "fidelity"),
        ("teff_gspphot", "teff_gspphot"),
        ("logg_gspphot", "logg_gspphot"),
        ("mh_gspphot", "mh_gspphot"),
        ("distance_gspphot", "distance_gspphot"),
        ("Parallax (mas)", "parallax"),
        ("Parallax err (mas)", "parallax_error"),
        ("PM RA (mas/yr)", "pmra"),
        ("PM Dec (mas/yr)", "pmdec"),
        ("Total PM (mas/yr)", "pm_total"),
        ("high_pm_flag", "high_pm_flag"),
    ]),
    ("Photometry", [
        ("2MASS J", "tmass_j"),
        ("2MASS J err", "tmass_j_err"),
        ("2MASS H", "tmass_h"),
        ("2MASS H err", "tmass_h_err"),
        ("2MASS K", "tmass_k"),
        ("2MASS K err", "tmass_k_err"),
        ("unWISE W1", "unwise_w1"),
        ("unWISE W1 err", "unwise_w1_err"),
        ("unWISE W2", "unwise_w2"),
        ("unWISE W2 err", "unwise_w2_err"),
        ("H-K", "H_K"),
        ("H-K (dered)", "H_K_dered"),
        ("W1-W2", "W1_W2"),
        ("W1-W2 (dered)", "W1_W2_dered"),
        ("BP-RP (dered)", "bprp0"),
        ("IPHAS H-alpha", "iphas_ha_mag"),
        ("IPHAS R-I", "iphas_r_i"),
        ("IPHAS H-alpha excess", "iphas_ha_excess"),
        ("unWISE W1 z-score", "unwise_w1_zscore"),
        ("unWISE W2 z-score", "unwise_w2_zscore"),
    ]),
    ("Galactic Coordinates", [
        ("Galactic l", "gal_l"),
        ("Galactic b", "gal_b"),
    ]),
    ("StarHorse", [
        ("Distance (kpc)", "dist50"),
        ("Distance 16th", "dist16"),
        ("Distance 84th", "dist84"),
        ("Teff", "teff50"),
        ("Teff 16th", "teff16"),
        ("Teff 84th", "teff84"),
        ("log g", "logg50"),
        ("log g 16th", "logg16"),
        ("log g 84th", "logg84"),
        ("[M/H]", "met50"),
        ("[M/H] 16th", "met16"),
        ("[M/H] 84th", "met84"),
        ("Mass 16th", "mass16"),
        ("Mass 84th", "mass84"),
        ("A_V", "av50"),
        ("A_V 16th", "av16"),
        ("A_V 84th", "av84"),
        ("A_G", "ag50"),
        ("A_BP", "abp50"),
        ("A_RP", "arp50"),
        ("M_G (dered)", "mg0"),
    ]),
    ("Phase Folding", [
        ("Period (d)", "phase_period_days"),
        ("Source", "phase_source"),
        ("Plot ready", "phase_plot_ready"),
        ("Quality score", "phase_quality_score"),
    ]),
    ("External Follow-up", [
        ("Has spectrum", "has_spectrum"),
        ("Spectrum sources", "spectrum_sources"),
        ("Spectrum links", "spectrum_links"),
        ("ATLAS photometry", "atlas_has_phot"),
        ("ATLAS cyan n", "atlas_n_det_cyan"),
        ("ATLAS orange n", "atlas_n_det_orange"),
        ("ATLAS cyan range", "atlas_cyan_range"),
        ("ATLAS orange range", "atlas_orange_range"),
        ("NEOWISE epochs", "neowise_n_epochs"),
        ("NEOWISE W1 range", "neowise_w1_range"),
        ("NEOWISE W2 range", "neowise_w2_range"),
    ]),
    ("Extinction & Environment", [
        ("A_v_3d", "A_v_3d"),
        ("ebv_3d", "ebv_3d"),
        ("Dust max dist (kpc)", "dust_max_dist_kpc"),
        ("population", "population"),
        ("age50", "age50"),
        ("mass50", "mass50"),
        ("banyan_field_prob", "banyan_field_prob"),
        ("banyan_best_assoc", "banyan_best_assoc"),
        ("R_gal (kpc)", "rgal"),
        ("X_gal", "xgal"),
        ("Y_gal", "ygal"),
        ("Z_gal", "zgal"),
    ]),
    ("YSO / Classification", [
        ("yso_class", "yso_class"),
        ("Trigger", "trigger_type"),
        ("final_class", "final_class"),
        ("P_eb", "P_eb"),
        ("P_cv", "P_cv"),
        ("P_starspot", "P_starspot"),
        ("P_disk", "P_disk"),
        ("a_circ_au", "a_circ_au"),
        ("transit_prob", "transit_prob"),
        ("hill_radius_rsun", "hill_radius_rsun"),
    ]),
    ("Filter Flags", [
        ("failed_any", "failed_any"),
        ("failed_sparse", "failed_sparse"),
        ("failed_multi_camera", "failed_multi_camera"),
        ("failed_vsx", "failed_vsx"),
        ("failed_evidence_strength", "failed_evidence_strength"),
        ("failed_run_robustness", "failed_run_robustness"),
        ("failed_morphology", "failed_morphology"),
        ("failed_score", "failed_score"),
        ("failed_periodicity", "failed_periodicity"),
        ("failed_gaia_ruwe", "failed_gaia_ruwe"),
        ("failed_gaia_pm", "failed_gaia_pm"),
        ("failed_periodic_catalog", "failed_periodic_catalog"),
        ("failed_signal_amplitude", "failed_signal_amplitude"),
        ("Bad cameras filtered", "bad_cameras_filtered"),
    ]),
]

# Flat list derived from grouped metadata fields.
REVIEW_METADATA_FIELDS: list[tuple[str, str]] = [
    (label, key)
    for _group_name, fields in REVIEW_METADATA_GROUPS
    for label, key in fields
]

# Groups that start expanded in the Dash GUI.
_DEFAULT_OPEN_GROUPS: set[str] = {
    "Identification",
    "Crossmatch",
    "Vetting",
    "External Follow-up",
    "Periodicity",
    "Dip Detection",
    "Jump Detection",
    "Stellar Parameters",
    "Extinction & Environment",
    "YSO / Classification",
}


def normalize_vsx_record(record: dict[str, Any]) -> dict[str, Any]:
    return dict(record)


def normalize_vsx_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "vsx_class" not in out.columns:
        out["vsx_class"] = pd.NA

    if "vsx_sep_arcsec" not in out.columns:
        out["vsx_sep_arcsec"] = pd.NA
    return out


def _is_present(val: Any) -> bool:
    """Return True if *val* should be shown (not None/NaN/empty)."""
    if val is None:
        return False
    if isinstance(val, float) and pd.isna(val):
        return False
    if isinstance(val, str) and not val.strip():
        return False
    return True


def _format_large_integer_like(val: Any) -> Any:
    """Return non-scientific integer string when *val* is integer-like.

    Gaia IDs often arrive as float/scientific-notation strings (e.g. "4.27e+17").
    For display we normalize those to full integer strings.
    """
    if val is None:
        return val
    s = str(val).strip()
    if not s:
        return val
    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError):
        return val
    if d != d.to_integral_value():
        return val
    try:
        return format(d.to_integral_value(), "f")
    except Exception:
        return val


def _format_value_for_display(key: str, val: Any) -> Any:
    if key == "gaia_id":
        return _format_large_integer_like(val)
    return val


def extract_review_metadata(
    payload: dict[str, Any],
) -> list[tuple[str, Any]]:
    """Return a flat list of ``(label, value)`` pairs for display.

    This is the original flat interface used by callers that don't need
    group structure (e.g. ``plot_batch.py``).
    """
    p = normalize_vsx_record(payload)
    rows: list[tuple[str, Any]] = []
    for label, key in REVIEW_METADATA_FIELDS:
        val = p.get(key)
        if not _is_present(val):
            continue
        val = _format_value_for_display(key, val)
        rows.append((label, val))
    return rows


def extract_review_metadata_grouped(
    payload: dict[str, Any],
) -> list[tuple[str, list[tuple[str, Any]]]]:
    """Return metadata organised by group.

    Returns a list of ``(group_name, items)`` tuples where *items* is a
    list of ``(label, value)`` pairs.  Groups with no present values are
    omitted.
    """
    p = normalize_vsx_record(payload)
    groups: list[tuple[str, list[tuple[str, Any]]]] = []
    for group_name, fields in REVIEW_METADATA_GROUPS:
        items: list[tuple[str, Any]] = []
        for label, key in fields:
            val = p.get(key)
            if not _is_present(val):
                continue
            val = _format_value_for_display(key, val)
            items.append((label, val))
        if items:
            groups.append((group_name, items))
    return groups


def is_group_default_open(group_name: str) -> bool:
    """Whether *group_name* should start expanded in the Dash GUI."""
    return group_name in _DEFAULT_OPEN_GROUPS


# ---------------------------------------------------------------------------
# External lookup URL construction
# ---------------------------------------------------------------------------

def _safe_float(val: Any) -> float | None:
    """Return *val* as a float if finite, else ``None``."""
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _safe_str(val: Any) -> str | None:
    """Return *val* as a non-empty stripped string, or ``None``."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    s = str(val).strip()
    return s if s else None


def build_external_lookup_links(
    payload: dict[str, Any],
) -> list[tuple[str, str]]:
    """Build ``(label, url)`` pairs for external astronomical services.

    Uses identifier-based URLs when a catalogue ID is available, falling
    back to coordinate-based (RA/Dec) cone searches otherwise.  Links
    whose required fields are missing are silently omitted.
    """
    links: list[tuple[str, str]] = []

    ra = _safe_float(payload.get("ra"))
    if ra is None:
        ra = _safe_float(payload.get("ra_deg"))

    dec = _safe_float(payload.get("dec"))
    if dec is None:
        dec = _safe_float(payload.get("dec_deg"))

    has_coords = ra is not None and dec is not None

    # -- SIMBAD ---------------------------------------------------------------
    simbad_id = _safe_str(payload.get("simbad_main_id"))
    if simbad_id:
        links.append((
            "SIMBAD",
            f"https://simbad.u-strasbg.fr/simbad/sim-id?Ident={quote_plus(simbad_id)}",
        ))
    elif has_coords:
        links.append((
            "SIMBAD",
            f"https://simbad.u-strasbg.fr/simbad/sim-coo?Coord={ra}+{dec}&CooFrame=FK5&CooEpoch=2000&Radius=10&Radius.unit=arcsec",
        ))

    # -- VizieR ---------------------------------------------------------------
    if has_coords:
        links.append((
            "VizieR",
            f"https://vizier.cds.unistra.fr/viz-bin/VizieR-4?-c={ra}+{dec}&-c.rs=10",
        ))

    # -- ADS ------------------------------------------------------------------
    if simbad_id:
        links.append((
            "ADS",
            f"https://ui.adsabs.harvard.edu/search/q=object%3A%22{quote(simbad_id)}%22",
        ))
    elif has_coords:
        links.append((
            "ADS",
            f"https://ui.adsabs.harvard.edu/search/q=pos(%22{ra}+{dec}%22%2C+10s)",
        ))

    # -- ASAS-SN Sky Patrol ---------------------------------------------------
    if has_coords:
        links.append((
            "Sky Patrol",
            f"https://asas-sn.osu.edu/sky-patrol/coordinate/{ra}/{dec}",
        ))

    # -- ALeRCE Explorer ------------------------------------------------------
    alerce_oid = _safe_str(payload.get("alerce_oid"))
    if alerce_oid:
        links.append((
            "ALeRCE",
            f"https://alerce.online/object/{quote(alerce_oid)}",
        ))

    # -- Gaia Archive ---------------------------------------------------------
    gaia_id = _safe_str(payload.get("gaia_id"))
    if gaia_id:
        gaia_id_norm = _format_large_integer_like(gaia_id)
        links.append((
            "Gaia",
            f"https://gea.esac.esa.int/archive/#sourceId={gaia_id_norm}",
        ))

    # -- TNS ------------------------------------------------------------------
    tns_name = _safe_str(payload.get("tns_name"))
    if tns_name:
        # TNS names are like "AT2021abc" or "SN2020xyz"; strip leading
        # prefixes that the URL path doesn't expect.
        tns_slug = tns_name.strip()
        links.append((
            "TNS",
            f"https://www.wis-tns.org/object/{quote(tns_slug)}",
        ))

    # -- ZTF / IRSA -----------------------------------------------------------
    if has_coords:
        links.append((
            "ZTF",
            f"https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?catalog=ztf_objects_dr22&objstr={ra}+{dec}&radius=5&radunits=arcsec",
        ))

    return links
