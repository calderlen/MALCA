"""Dependency-light taxonomy state for the terminal review interface.

This module deliberately contains no curses or persistence code.  It translates
the canonical review taxonomy into immutable menu data and owns the small,
mutable draft that a TUI edits before saving.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Optional, Tuple

from .classification_labels import resolve_catalog_class
from .taxonomy import (
    classification_confidence_from_score,
    json_list,
    keyboard_payload,
    selection_from_review,
)


@dataclass(frozen=True)
class MenuItem:
    """One immutable keyboard-selectable taxonomy option."""

    key: str
    value: str
    label: str


@dataclass(frozen=True)
class DetailSection:
    """One vertically rendered group in the portrait candidate-details pane.

    ``key`` is deliberately stable and lowercase so the curses layer can style
    individual groups without depending on their user-facing titles.  ``lines``
    are already width-bounded and contain no terminal-control sequences.
    """

    key: str
    title: str
    lines: Tuple[str, ...]


# The order mirrors the hierarchy: gate, primary morphology, then parent-gated
# subtype scores.
ML_CLASS_SCORE_FIELDS: Tuple[tuple[str, str, str], ...] = (
    (
        "prob_hierarchical_artifact_or_nonvariable",
        "P(reject)",
        "Reject",
    ),
    ("prob_usable_astrophysical_variable", "P(usable)", "Usable"),
    ("prob_dipper_dimming", "P(dip)", "Dipper"),
    ("prob_eb_geometric_periodic", "P(EB)", "EB"),
    ("prob_long_timescale_variable", "P(long)", "Long"),
    ("prob_brightening_transient", "P(bright)", "Bright"),
    ("prob_other_structured_variable", "P(other)", "Other"),
    ("prob_quasi_periodic_hierarchical", "P(QP)", "QP"),
    ("prob_microlensing_hierarchical", "P(micro)", "Micro"),
    ("prob_long_period_variable_hierarchical", "P(LPV)", "LPV"),
    ("prob_long_term_variable_hierarchical", "P(LTV)", "LTV"),
    ("prob_recurrent_dipper_hierarchical", "P(recur)", "Recur"),
    ("prob_single_dipper_hierarchical", "P(single)", "Single"),
)


def _menu_items(payload_items: Any) -> Tuple[MenuItem, ...]:
    return tuple(
        MenuItem(
            key=str(item["key"]).lower(),
            value=str(item["value"]),
            label=str(item["label"]),
        )
        for item in payload_items
    )


def _index_items(
    items: Tuple[MenuItem, ...], attribute: str
) -> Mapping[str, MenuItem]:
    return MappingProxyType({str(getattr(item, attribute)): item for item in items})


_KEYBOARD_PAYLOAD = keyboard_payload()

MORPHOLOGY_PRIMARY_ITEMS: Tuple[MenuItem, ...] = _menu_items(
    _KEYBOARD_PAYLOAD["morphology_primary"]
)
MORPHOLOGY_PRIMARY_BY_KEY: Mapping[str, MenuItem] = _index_items(
    MORPHOLOGY_PRIMARY_ITEMS, "key"
)
MORPHOLOGY_PRIMARY_BY_VALUE: Mapping[str, MenuItem] = _index_items(
    MORPHOLOGY_PRIMARY_ITEMS, "value"
)

PHYSICAL_PRIMARY_ITEMS: Tuple[MenuItem, ...] = _menu_items(
    _KEYBOARD_PAYLOAD["physical_primary"]
)
PHYSICAL_PRIMARY_BY_KEY: Mapping[str, MenuItem] = _index_items(
    PHYSICAL_PRIMARY_ITEMS, "key"
)
PHYSICAL_PRIMARY_BY_VALUE: Mapping[str, MenuItem] = _index_items(
    PHYSICAL_PRIMARY_ITEMS, "value"
)

_physical_secondary_items = {
    str(primary): _menu_items(items)
    for primary, items in _KEYBOARD_PAYLOAD["physical_secondary"].items()
}
PHYSICAL_SECONDARY_ITEMS: Mapping[str, Tuple[MenuItem, ...]] = MappingProxyType(
    _physical_secondary_items
)
PHYSICAL_SECONDARY_BY_KEY: Mapping[str, Mapping[str, MenuItem]] = MappingProxyType(
    {
        primary: _index_items(items, "key")
        for primary, items in _physical_secondary_items.items()
    }
)
PHYSICAL_SECONDARY_BY_VALUE: Mapping[str, Mapping[str, MenuItem]] = (
    MappingProxyType(
        {
            primary: _index_items(items, "value")
            for primary, items in _physical_secondary_items.items()
        }
    )
)

_secondary_items = {
    str(primary): _menu_items(items)
    for primary, items in _KEYBOARD_PAYLOAD["morphology_secondary"].items()
}
MORPHOLOGY_SECONDARY_ITEMS: Mapping[str, Tuple[MenuItem, ...]] = MappingProxyType(
    _secondary_items
)
MORPHOLOGY_SECONDARY_BY_KEY: Mapping[str, Mapping[str, MenuItem]] = MappingProxyType(
    {
        primary: _index_items(items, "key")
        for primary, items in _secondary_items.items()
    }
)
MORPHOLOGY_SECONDARY_BY_VALUE: Mapping[str, Mapping[str, MenuItem]] = (
    MappingProxyType(
        {
            primary: _index_items(items, "value")
            for primary, items in _secondary_items.items()
        }
    )
)


def primary_item_for_key(key: str) -> Optional[MenuItem]:
    """Return the primary-morphology item bound to ``key``, if any."""

    return MORPHOLOGY_PRIMARY_BY_KEY.get(str(key).lower())


def physical_primary_item_for_key(key: str) -> Optional[MenuItem]:
    """Return the broad physical-hypothesis item bound to ``key``, if any."""

    return PHYSICAL_PRIMARY_BY_KEY.get(str(key).lower())


def physical_secondary_items_for(primary: Optional[str]) -> Tuple[MenuItem, ...]:
    """Return the ordered subtype menu for a physical family."""

    if not primary:
        return ()
    return PHYSICAL_SECONDARY_ITEMS.get(str(primary), ())


def physical_secondary_item_for_key(
    primary: Optional[str], key: str
) -> Optional[MenuItem]:
    """Return the physical subtype bound to ``key`` within ``primary``."""

    if not primary:
        return None
    return PHYSICAL_SECONDARY_BY_KEY.get(str(primary), {}).get(str(key).lower())


def secondary_items_for(primary: Optional[str]) -> Tuple[MenuItem, ...]:
    """Return the ordered subtype menu for a primary morphology."""

    if not primary:
        return ()
    return MORPHOLOGY_SECONDARY_ITEMS.get(str(primary), ())


def secondary_item_for_key(primary: Optional[str], key: str) -> Optional[MenuItem]:
    """Return the subtype item bound to ``key`` within ``primary``."""

    if not primary:
        return None
    return MORPHOLOGY_SECONDARY_BY_KEY.get(str(primary), {}).get(str(key).lower())


@dataclass(frozen=True)
class DraftState:
    """Immutable snapshot used for cheap, deterministic dirty tracking."""

    morphology_primary: Optional[str]
    morphology_secondaries: Tuple[str, ...]
    physical_primary: Optional[str]
    physical_secondary: Optional[str]
    confidence: Optional[int]
    needs_followup: bool
    notes: str


@dataclass
class ReviewDraft:
    """The small portion of a review edited by the terminal reviewer."""

    morphology_primary: Optional[str] = None
    morphology_secondaries: list[str] = field(default_factory=list)
    physical_primary: Optional[str] = None
    physical_secondary: Optional[str] = None
    confidence: Optional[int] = None
    needs_followup: bool = False
    notes: str = ""
    _saved_state: DraftState = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.morphology_primary = _optional_text(self.morphology_primary)
        self.morphology_secondaries = _ordered_unique(self.morphology_secondaries)
        self.physical_primary = _optional_text(self.physical_primary)
        self.physical_secondary = _optional_text(self.physical_secondary)
        self.confidence = classification_confidence_from_score(self.confidence)
        self.needs_followup = bool(self.needs_followup)
        self.notes = str(self.notes or "")
        self._saved_state = self.snapshot()

    @classmethod
    def from_review(cls, review: Mapping[str, Any]) -> "ReviewDraft":
        """Build a clean draft from an existing review mapping.

        ``selection_from_review`` is intentionally used here so legacy scalar
        subtypes and ``morphology_secondary_json`` retain the same precedence,
        de-duplication, and ordering as the browser reviewer.
        """

        source = dict(review)
        selection = selection_from_review(source)
        if source.get("needs_followup") is None:
            status = source.get("workflow_status") or source.get("status")
            needs_followup = str(status or "").strip().lower() == "needs_followup"
        else:
            needs_followup = _as_bool(source.get("needs_followup"))
        return cls(
            morphology_primary=selection.get("morphology_primary"),
            morphology_secondaries=list(
                selection.get("morphology_secondary_list") or []
            ),
            physical_primary=selection.get("physical_primary"),
            physical_secondary=selection.get("physical_secondary"),
            confidence=selection.get("classification_confidence"),
            needs_followup=needs_followup,
            notes=str(source.get("notes") or ""),
        )

    def snapshot(self) -> DraftState:
        """Return an immutable representation of the current editable state."""

        return DraftState(
            morphology_primary=self.morphology_primary,
            morphology_secondaries=tuple(self.morphology_secondaries),
            physical_primary=self.physical_primary,
            physical_secondary=self.physical_secondary,
            confidence=self.confidence,
            needs_followup=self.needs_followup,
            notes=self.notes,
        )

    @property
    def dirty(self) -> bool:
        """Whether the draft differs from the last loaded or saved state."""

        return self.snapshot() != self._saved_state

    @property
    def morphology_secondary(self) -> Optional[str]:
        """Legacy scalar subtype: the first selected subtype, if present."""

        return self.morphology_secondaries[0] if self.morphology_secondaries else None

    @property
    def morphology_secondary_json(self) -> str:
        """Canonical compact JSON representation used by the review store."""

        return json_list(self.morphology_secondaries)

    def select_primary(self, primary: str) -> bool:
        """Select a primary morphology and return whether it changed.

        Subtypes are cleared only on an actual primary change.  Menu-mode
        transitions belong to the TUI and are deliberately not represented here.
        """

        value = str(primary)
        if value not in MORPHOLOGY_PRIMARY_BY_VALUE:
            raise ValueError(f"Unknown morphology primary: {value!r}")
        changed = value != self.morphology_primary
        if changed:
            self.morphology_primary = value
            self.morphology_secondaries.clear()
        return changed

    def toggle_subtype(self, subtype: str) -> bool:
        """Toggle a subtype and return ``True`` when it is selected afterward."""

        if not self.morphology_primary:
            raise ValueError("Select a morphology primary before selecting a subtype")
        value = str(subtype)
        allowed = MORPHOLOGY_SECONDARY_BY_VALUE.get(self.morphology_primary, {})
        if value not in allowed:
            raise ValueError(
                f"Subtype {value!r} does not belong to morphology primary "
                f"{self.morphology_primary!r}"
            )
        if value in self.morphology_secondaries:
            self.morphology_secondaries.remove(value)
            return False
        self.morphology_secondaries.append(value)
        return True

    def clear_subtypes(self) -> bool:
        """Clear all selected subtypes and return whether anything changed."""

        changed = bool(self.morphology_secondaries)
        self.morphology_secondaries.clear()
        return changed

    def select_physical_primary(self, physical_primary: Optional[str]) -> bool:
        """Select or clear the broad physical hypothesis.

        As in the browser reviewer, changing the family clears the old subtype.
        """

        value = _optional_text(physical_primary)
        if value is not None and value not in PHYSICAL_PRIMARY_BY_VALUE:
            raise ValueError(f"Unknown physical primary: {value!r}")
        changed = value != self.physical_primary
        if changed:
            self.physical_primary = value
            self.physical_secondary = None
        return changed

    def toggle_physical_subtype(self, physical_secondary: str) -> bool:
        """Toggle the one browser-compatible subtype for the selected family."""

        if not self.physical_primary:
            raise ValueError(
                "Select a physical hypothesis before selecting a physical subtype"
            )
        value = str(physical_secondary)
        allowed = PHYSICAL_SECONDARY_BY_VALUE.get(self.physical_primary, {})
        if value not in allowed:
            raise ValueError(
                f"Physical subtype {value!r} does not belong to "
                f"{self.physical_primary!r}"
            )
        if value == self.physical_secondary:
            self.physical_secondary = None
            return False
        self.physical_secondary = value
        return True

    def clear_physical_subtype(self) -> bool:
        """Clear the physical subtype and return whether it changed."""

        changed = self.physical_secondary is not None
        self.physical_secondary = None
        return changed

    def set_notes(self, notes: Any) -> None:
        """Replace the free-form review notes."""

        self.notes = str(notes or "")

    def set_confidence(self, confidence: Any) -> None:
        """Set label confidence, rejecting values outside the canonical 1--4."""

        score = classification_confidence_from_score(confidence)
        if score is None:
            raise ValueError("Confidence must be an integer from 1 to 4")
        self.confidence = score

    def toggle_followup(self) -> bool:
        """Toggle and return the new follow-up state."""

        self.needs_followup = not self.needs_followup
        return self.needs_followup

    def validate(self) -> Tuple[str, ...]:
        """Return user-facing save errors for required review fields."""

        errors = []
        if self.morphology_primary not in MORPHOLOGY_PRIMARY_BY_VALUE:
            errors.append("Morphology is required")
        if classification_confidence_from_score(self.confidence) is None:
            errors.append("Confidence must be from 1 to 4")
        if (
            self.physical_primary is not None
            and self.physical_primary not in PHYSICAL_PRIMARY_BY_VALUE
        ):
            errors.append("Physical hypothesis is not recognized")
        if self.physical_secondary is not None:
            allowed = PHYSICAL_SECONDARY_BY_VALUE.get(self.physical_primary or "", {})
            if self.physical_secondary not in allowed:
                errors.append("Physical subtype does not match its hypothesis")
        return tuple(errors)

    def mark_saved(self) -> None:
        """Make the current state the baseline for dirty tracking."""

        self._saved_state = self.snapshot()

    def to_selection(self) -> dict[str, Any]:
        """Return the canonical morphology/confidence fields for persistence."""

        secondaries = list(self.morphology_secondaries)
        return {
            "morphology_primary": self.morphology_primary,
            "morphology_secondary": secondaries[0] if secondaries else None,
            "morphology_secondary_list": secondaries,
            "morphology_secondary_json": json_list(secondaries),
            "physical_primary": self.physical_primary,
            "physical_secondary": self.physical_secondary,
            "classification_confidence": self.confidence,
        }


def compact_detail_lines(
    payload: Optional[Mapping[str, Any]],
    *,
    phase_period_days: Any = None,
    phase_period_source: Any = None,
) -> Tuple[str, ...]:
    """Format high-value candidate metadata for the TUI details pane.

    The helper is deliberately pure and terminal-library agnostic.  Explicit
    phase values win because the image coordinator may have refined the stored
    payload period with a harmonic check or an automatic search.
    """

    values: Mapping[str, Any] = payload or {}
    ruwe = _first_finite(values, "ruwe", "ruwe_gaia")
    class_scores = tuple(
        (short_label, _format_probability(_first_finite(values, column)))
        for column, _detail_label, short_label in ML_CLASS_SCORE_FIELDS
    )
    q_value = _first_finite(
        values,
        "stats_variability_quasi_periodicity_q",
        "quasi_periodicity_q",
        "q_stat",
    )
    m_value = _first_finite(
        values,
        "stats_variability_flux_asymmetry_m",
        "flux_asymmetry_m",
        "m_stat",
    )

    period = _finite_number(phase_period_days)
    if period is None or period <= 0:
        period = _first_positive_finite(
            values,
            "phase_period_days",
            "period_consensus_days",
            "pre_periodicity_selected_period",
            "periodicity_period",
            "pdm_corrected_period",
            "ce_corrected_period",
            "pdm_period",
            "ce_period",
        )
    source = _optional_text(phase_period_source)
    if source is None:
        source = _first_text(values, "phase_source", "period_primary_source")

    vsx = _first_text(values, "vsx_class", "period_vsx_class") or "—"
    gaia_var = _first_text(values, "gaia_var_class")
    if gaia_var is None:
        gaia_var = _tri_state_text(values, "gaia_var_flag")
    likely_known = _tri_state_text(values, "vetting_likely_known")

    period_text = _format_period(period)
    if source:
        period_text += f" ({source})"
    return (
        "ML " + "   ".join(f"{label} {score}" for label, score in class_scores[:2]),
        "ML " + "   ".join(f"{label} {score}" for label, score in class_scores[2:5]),
        "ML " + "   ".join(f"{label} {score}" for label, score in class_scores[5:7]),
        "ML " + "   ".join(f"{label} {score}" for label, score in class_scores[7:10]),
        "ML " + "   ".join(f"{label} {score}" for label, score in class_scores[10:]),
        f"Q {_format_probability(q_value)}   M {_format_probability(m_value)}   period {period_text}",
        f"RUWE {_format_number(ruwe, 2)}",
        f"VSX {vsx}   Gaia VAR {gaia_var}   likely known {likely_known}",
    )


def detail_sections(
    payload: Optional[Mapping[str, Any]],
    *,
    phase_period_days: Any = None,
    phase_period_source: Any = None,
    width: Optional[int] = None,
) -> Tuple[DetailSection, ...]:
    """Return portrait-oriented candidate details grouped by purpose.

    Unlike :func:`compact_detail_lines`, this representation favors vertical
    space over horizontal space.  That makes it suitable for a narrow, tall
    terminal while retaining explicit section boundaries for curses styling.
    ``width`` limits every returned line (including its two-space indent); it
    may be omitted when the caller already clips output at the window edge.

    Explicit phase values have the same precedence as
    :func:`compact_detail_lines`, ensuring the terminal metadata agrees with
    the currently displayed phase-fold image.
    """

    values: Mapping[str, Any] = payload or {}
    ruwe = _first_finite(values, "ruwe", "ruwe_gaia")
    pm_total = _pm_total_masyr(values)
    pm_text = _format_number(pm_total, 1) if pm_total is not None else "—"
    # SED slope (α ≡ d log(λF_λ)/d log λ over the optical/near-IR baseline).
    # Surfacing it in the sidebar lets a reviewer triage YSO vs. reddened
    # main-sequence vs. featureless source without opening the diagnostics
    # panel.
    sed_alpha = _first_finite(values, "sed_alpha", "alpha_ir", "sed_slope")
    sed_alpha_class = _first_text(values, "sed_alpha_class")
    class_scores = tuple(
        (detail_label, _format_probability(_first_finite(values, column)))
        for column, detail_label, _short_label in ML_CLASS_SCORE_FIELDS
    )
    q_value = _first_finite(
        values,
        "stats_variability_quasi_periodicity_q",
        "quasi_periodicity_q",
        "q_stat",
    )
    m_value = _first_finite(
        values,
        "stats_variability_flux_asymmetry_m",
        "flux_asymmetry_m",
        "m_stat",
    )

    period = _finite_number(phase_period_days)
    if period is None or period <= 0:
        period = _first_positive_finite(
            values,
            "phase_period_days",
            "period_consensus_days",
            "pre_periodicity_selected_period",
            "periodicity_period",
            "pdm_corrected_period",
            "ce_corrected_period",
            "pdm_period",
            "ce_period",
        )
    source = _optional_text(phase_period_source)
    if source is None:
        source = _first_text(values, "phase_source", "period_primary_source")

    vsx = _first_text(values, "vsx_class", "period_vsx_class") or "—"
    gaia_var = _first_text(values, "gaia_var_class")
    if gaia_var is None:
        gaia_var = _tri_state_text(values, "gaia_var_flag")
    likely_known = _tri_state_text(values, "vetting_likely_known")
    simbad = _format_simbad_catalog_text(values)
    asassn_variable = _format_asassn_catalog_text(values)

    if sed_alpha is not None:
        alpha_text = f"{sed_alpha:+.2f}"
    else:
        alpha_text = "—"
    alpha_class_text = sed_alpha_class or "—"

    teff = _first_finite(values, "teff50")
    teff_lo = _first_finite(values, "teff16")
    teff_hi = _first_finite(values, "teff84")
    teff_text = _format_asymmetric(
        teff,
        teff_lo,
        teff_hi,
        precision=0,
    )
    logg = _first_finite(values, "logg50")
    logg_lo = _first_finite(values, "logg16")
    logg_hi = _first_finite(values, "logg84")
    logg_text = _format_asymmetric(
        logg,
        logg_lo,
        logg_hi,
        precision=2,
    )
    met_text = _format_asymmetric(
        _first_finite(values, "met50"),
        _first_finite(values, "met16"),
        _first_finite(values, "met84"),
        precision=2,
    )
    mass_text = _format_asymmetric(
        _first_finite(values, "mass50"),
        _first_finite(values, "mass16"),
        _first_finite(values, "mass84"),
        precision=2,
    )
    av_text = _format_asymmetric(
        _first_finite(values, "av50"),
        _first_finite(values, "av16"),
        _first_finite(values, "av84"),
        precision=2,
    )

    assoc = _first_text(values, "banyan_best_assoc") or "—"
    assoc_prob = _first_finite(values, "banyan_best_assoc_prob")
    field_prob = _first_finite(values, "banyan_field_prob")
    if assoc != "—" and assoc_prob is not None:
        banyan_text = f"{assoc} {assoc_prob:.2f}"
    else:
        banyan_text = assoc
    if field_prob is not None:
        banyan_text = f"{banyan_text}  field {field_prob:.2f}"

    eb_level = _first_text(values, "gaia_eb_evidence_level")
    eb_score = _first_finite(values, "gaia_eb_evidence_score")
    eb_period = _first_finite(values, "gaia_eb_period")
    eb_families = _first_text(values, "gaia_binary_evidence_families")
    if eb_level and eb_level.lower() not in {"none", "nan"}:
        eb_text = eb_level.replace("_", " ")
        if eb_score is not None:
            eb_text = f"{eb_text} ({eb_score:.2f})"
    else:
        eb_text = "none"
    if eb_period is not None and eb_period > 0:
        eb_text = f"{eb_text}  P={eb_period:.4g}d"
    if eb_families and eb_families.lower() not in {"none", "nan"}:
        families = eb_families.replace(",", "+")
        if len(families) > 28:
            families = families[:27] + "…"
        eb_text = f"{eb_text}  {families}"

    mean_mag = _first_finite(
        values,
        "stats_clipped_mean_mag_3sigma_about_median",
        "clipped_mean_mag_3sigma_about_median",
    )
    mean_err = _first_finite(
        values,
        "stats_clipped_std_mag_3sigma_about_median",
        "clipped_std_mag_3sigma_about_median",
    )
    if mean_mag is not None and mean_err is not None:
        mean_text = f"{mean_mag:.3f}±{mean_err:.3f}"
    elif mean_mag is not None:
        mean_text = f"{mean_mag:.3f}"
    else:
        mean_text = "—"

    return (
        _detail_section(
            "ml_class_scores",
            "ML CLASS SCORES",
            class_scores,
            width=width,
        ),
        _detail_section(
            "signal",
            "SIGNAL",
            (
                ("Q", _format_probability(q_value)),
                ("M", _format_probability(m_value)),
                ("mean mag", mean_text),
                ("period", _format_period(period)),
                ("source", source or "—"),
            ),
            width=width,
        ),
        _detail_section(
            "astrometry",
            "ASTROMETRY",
            (
                ("RUWE", _format_number(ruwe, 2)),
                ("PM", pm_text),
                ("α_SED", alpha_text),
                ("α class", alpha_class_text),
            ),
            width=width,
        ),
        _detail_section(
            "starhorse",
            "STARHORSE",
            (
                ("Teff", teff_text),
                (
                    "type",
                    _approximate_stellar_type(
                        teff,
                        teff_lo,
                        teff_hi,
                        logg,
                        logg_lo,
                        logg_hi,
                    ),
                ),
                ("log g", logg_text),
                ("[M/H]", met_text),
                ("Mass", mass_text),
                ("A_V", av_text),
            ),
            width=width,
        ),
        _detail_section(
            "context",
            "CONTEXT",
            (
                ("BANYAN", banyan_text),
                ("EB", eb_text),
            ),
            width=width,
        ),
        _detail_section(
            "catalogs",
            "CATALOGS",
            (
                ("VSX", vsx),
                ("Gaia VAR", gaia_var),
                ("ASAS-SN", asassn_variable),
                ("SIMBAD", simbad),
                ("known", likely_known),
            ),
            width=width,
        ),
    )


def _format_asymmetric(
    value: Optional[float],
    lo: Optional[float],
    hi: Optional[float],
    *,
    precision: int = 0,
) -> str:
    """Compact symmetric uncertainty from StarHorse-style percentiles."""
    if value is None:
        return "—"
    if precision <= 0 and abs(value) >= 10:
        center = f"{value:.0f}"
        err_fmt = "{:.0f}"
    else:
        digits = max(precision, 2)
        center = f"{value:.{digits}f}"
        err_fmt = f"{{:.{digits}f}}"
    if lo is None or hi is None:
        return center
    down = max(0.0, value - lo)
    up = max(0.0, hi - value)
    err = max(up, down)
    return f"{center}±{err_fmt.format(err)}"


_DWARF_SPECTRAL_TYPE_TEMPERATURES: Tuple[Tuple[str, float], ...] = (
    ("O3", 44_900.0),
    ("O4", 42_900.0),
    ("O5", 41_400.0),
    ("O6", 39_500.0),
    ("O7", 37_100.0),
    ("O8", 35_100.0),
    ("O9", 33_300.0),
    ("B0", 31_500.0),
    ("B1", 26_000.0),
    ("B2", 20_600.0),
    ("B3", 17_000.0),
    ("B4", 16_000.0),
    ("B5", 15_000.0),
    ("B6", 14_000.0),
    ("B7", 13_000.0),
    ("B8", 11_900.0),
    ("B9", 10_700.0),
    ("A0", 9_700.0),
    ("A1", 9_200.0),
    ("A2", 8_840.0),
    ("A3", 8_550.0),
    ("A4", 8_270.0),
    ("A5", 8_080.0),
    ("A6", 8_000.0),
    ("A7", 7_800.0),
    ("A8", 7_500.0),
    ("A9", 7_440.0),
    ("F0", 7_220.0),
    ("F1", 7_030.0),
    ("F2", 6_820.0),
    ("F3", 6_750.0),
    ("F4", 6_670.0),
    ("F5", 6_550.0),
    ("F6", 6_350.0),
    ("F7", 6_280.0),
    ("F8", 6_180.0),
    ("F9", 6_050.0),
    ("G0", 5_930.0),
    ("G1", 5_860.0),
    ("G2", 5_770.0),
    ("G3", 5_720.0),
    ("G4", 5_680.0),
    ("G5", 5_660.0),
    ("G6", 5_600.0),
    ("G7", 5_550.0),
    ("G8", 5_480.0),
    ("G9", 5_380.0),
    ("K0", 5_270.0),
    ("K1", 5_170.0),
    ("K2", 5_100.0),
    ("K3", 4_830.0),
    ("K4", 4_600.0),
    ("K5", 4_440.0),
    ("K6", 4_300.0),
    ("K7", 4_100.0),
    ("K8", 3_990.0),
    ("K9", 3_930.0),
    ("M0", 3_850.0),
    ("M1", 3_660.0),
    ("M2", 3_560.0),
    ("M3", 3_430.0),
    ("M4", 3_210.0),
    ("M5", 3_060.0),
    ("M6", 2_810.0),
    ("M7", 2_680.0),
    ("M8", 2_570.0),
    ("M9", 2_380.0),
)


def _spectral_subtype_from_teff(teff_kelvin: float) -> str:
    """Return the nearest approximate dwarf-sequence temperature subtype."""
    return min(
        _DWARF_SPECTRAL_TYPE_TEMPERATURES,
        key=lambda item: abs(math.log(teff_kelvin) - math.log(item[1])),
    )[0]


def _luminosity_class_from_logg(logg: float) -> str:
    """Return a deliberately coarse luminosity class inferred from log(g)."""
    if logg >= 4.0:
        return "V"
    if logg >= 3.5:
        return "IV"
    if logg >= 2.5:
        return "III"
    if logg >= 1.5:
        return "II"
    return "I"


def _approximate_stellar_type(
    teff: Optional[float],
    teff_lo: Optional[float],
    teff_hi: Optional[float],
    logg: Optional[float],
    logg_lo: Optional[float],
    logg_hi: Optional[float],
) -> str:
    """Format a Teff subtype range and optional log(g) luminosity-class range.

    The result is an inference from model parameters, not an MK spectral
    classification. Teff and log(g) use their original asymmetric percentile
    bounds rather than the compact symmetric uncertainty printed by the TUI.
    """
    temperatures = [
        value
        for value in (teff, teff_lo, teff_hi)
        if value is not None and value > 0
    ]
    if not temperatures:
        return "—"

    hot_type = _spectral_subtype_from_teff(max(temperatures))
    cool_type = _spectral_subtype_from_teff(min(temperatures))
    subtype_text = (
        hot_type if hot_type == cool_type else f"{hot_type}–{cool_type}"
    )

    gravities = [
        value for value in (logg, logg_lo, logg_hi) if value is not None
    ]
    if not gravities:
        return f"≈{subtype_text}"
    bright_class = _luminosity_class_from_logg(min(gravities))
    faint_class = _luminosity_class_from_logg(max(gravities))
    luminosity_text = (
        bright_class
        if bright_class == faint_class
        else f"{bright_class}–{faint_class}"
    )
    return f"≈{subtype_text} {luminosity_text}"


def _detail_section(
    key: str,
    title: str,
    rows: Tuple[Tuple[str, str], ...],
    *,
    width: Optional[int],
) -> DetailSection:
    return DetailSection(
        key=key,
        title=title,
        lines=tuple(_format_detail_row(label, value, width) for label, value in rows),
    )


def _format_detail_row(label: str, value: str, width: Optional[int]) -> str:
    line = f"  {label:<9} {value}"
    if width is None:
        return line
    try:
        limit = max(1, int(width))
    except (TypeError, ValueError):
        return line
    if len(line) <= limit:
        return line
    if limit == 1:
        return "…"
    return line[: limit - 1].rstrip() + "…"


def _optional_text(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_finite(values: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        number = _finite_number(values.get(key))
        if number is not None:
            return number
    return None


def _first_positive_finite(
    values: Mapping[str, Any], *keys: str
) -> Optional[float]:
    for key in keys:
        number = _finite_number(values.get(key))
        if number is not None and number > 0:
            return number
    return None


def _first_text(values: Mapping[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        text = _optional_text(values.get(key))
        if text is not None and text.lower() not in {"nan", "none", "null"}:
            return text
    return None


def _tri_state_text(values: Mapping[str, Any], key: str) -> str:
    if key not in values or values.get(key) is None:
        return "—"
    value = values.get(key)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "nan", "none", "null"}:
            return "—"
        if normalized in {"0", "false", "no", "off"}:
            return "no"
        if normalized in {"1", "true", "yes", "on"}:
            return "yes"
    return "yes" if bool(value) else "no"


def _pm_total_masyr(values: Mapping[str, Any]) -> float | None:
    """Return total proper motion in mas/yr from payload columns."""

    pm_total = _first_finite(values, "pm_total")
    if pm_total is not None:
        return float(pm_total)
    pmra = _first_finite(values, "pmra", "pmra_gaia")
    pmdec = _first_finite(values, "pmdec", "pmdec_gaia")
    if pmra is None or pmdec is None:
        return None
    return float(math.hypot(float(pmra), float(pmdec)))


def _format_number(value: Optional[float], places: int) -> str:
    if value is None:
        return "—"
    return f"{value:.{places}f}"


def _format_probability(value: Optional[float]) -> str:
    if value is None:
        return "—"
    text = f"{value:.2f}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def _format_period_days(value: Optional[float]) -> str:
    if value is None or value <= 0:
        return "—"
    if value < 0.01:
        return f"{value:.3g}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_period(value: Optional[float]) -> str:
    if value is None or value <= 0:
        return "—"
    return f"{_format_period_days(value)} d"


_CATALOG_LABEL_SKIP = frozenset(
    {
        "",
        "—",
        "-",
        "none",
        "null",
        "nan",
        "unknown",
        "no",
        "yes",
        "0",
        "1",
        "false",
        "true",
        "off",
        "on",
    }
)


def _meaningful_catalog_text(value: object) -> Optional[str]:
    text = _optional_text(value)
    if text is None:
        return None
    if text.strip().lower() in _CATALOG_LABEL_SKIP:
        return None
    return text


def _catalog_class_short_label(column: str, value: object) -> str:
    """Match Dash vetting-banner catalog labels without the trailing source tag."""
    resolved = resolve_catalog_class(column, value)
    if not resolved.value:
        return ""
    suffix = f" [{resolved.source}]" if resolved.source else ""
    if suffix and resolved.label.endswith(suffix):
        return resolved.label[: -len(suffix)]
    return resolved.label


def _format_simbad_catalog_text(values: Mapping[str, Any]) -> str:
    otype = _first_text(values, "simbad_otype")
    if otype is not None:
        label = _catalog_class_short_label("simbad_otype", otype)
        return label or otype
    main_id = _first_text(values, "simbad_main_id")
    if main_id is not None:
        return main_id
    return "—"


def _format_asassn_catalog_text(values: Mapping[str, Any]) -> str:
    variable_type = _meaningful_catalog_text(
        _first_text(
            values,
            "asassn_var_type",
            "period_asassn_var_class",
            "asas_sn_var_type",
            "asas_var_type",
        )
    )
    variable_name = _meaningful_catalog_text(
        _first_text(values, "asassn_var_name")
    )
    text = variable_type or variable_name
    if text is None:
        return "—"
    period = _first_positive_finite(
        values,
        "asassn_var_period",
        "period_asassn_var_days",
        "period_asassn_var_period",
        "period_asassn_var_period_days",
    )
    if period is not None:
        text = f"{text} P={period:.4f}d"
    return text


def external_catalog_labels(payload: Optional[Mapping[str, Any]]) -> Tuple[str, ...]:
    """Return short external-catalog classification labels for display titles."""
    values: Mapping[str, Any] = payload or {}
    labels: list[str] = []

    vsx = _meaningful_catalog_text(
        _first_text(values, "vsx_class", "period_vsx_class")
    )
    if vsx is not None:
        labels.append(f"VSX {vsx}")

    gaia_var = _meaningful_catalog_text(_first_text(values, "gaia_var_class"))
    if gaia_var is not None:
        labels.append(f"Gaia {gaia_var}")

    simbad_otype = _first_text(values, "simbad_otype")
    if simbad_otype is not None:
        simbad = _meaningful_catalog_text(
            _catalog_class_short_label("simbad_otype", simbad_otype) or simbad_otype
        )
    else:
        simbad = _meaningful_catalog_text(_first_text(values, "simbad_main_id"))
    if simbad is not None:
        labels.append(f"SIMBAD {simbad}")

    asas_var = _meaningful_catalog_text(
        _first_text(
            values,
            "asassn_var_type",
            "period_asassn_var_class",
            "asas_sn_var_type",
            "asas_var_type",
        )
    )
    if asas_var is not None:
        labels.append(f"ASAS-SN {asas_var}")

    return tuple(labels)


def build_review_identity_line(
    payload: Optional[Mapping[str, Any]],
    *,
    asas_sn_id: Optional[str] = None,
) -> str:
    """Compact sidebar identity: ASAS-SN and Gaia IDs only."""
    values: Mapping[str, Any] = payload or {}
    parts: list[str] = []

    asas = _optional_text(asas_sn_id) or _first_text(values, "asas_sn_id")
    if asas is not None:
        parts.append(f"ASAS-SN: {asas}")

    gaia_id = _first_text(values, "gaia_id", "source_id")
    if gaia_id is not None:
        parts.append(f"GAIA: {gaia_id}")

    return "  ".join(parts) if parts else "MALCA Review"


def build_review_display_title(
    payload: Optional[Mapping[str, Any]],
    *,
    asas_sn_id: Optional[str] = None,
) -> str:
    """Compact review title: ASAS-SN ID, Gaia ID, and catalog classes only."""
    values: Mapping[str, Any] = payload or {}
    parts: list[str] = []

    asas = _optional_text(asas_sn_id) or _first_text(values, "asas_sn_id")
    if asas is not None:
        parts.append(f"ASAS-SN {asas}")

    gaia_id = _first_text(values, "gaia_id", "source_id")
    if gaia_id is not None:
        parts.append(f"Gaia {gaia_id}")

    parts.extend(external_catalog_labels(values))
    compact = [part for part in parts if part]
    return "  ·  ".join(compact) if compact else "MALCA Review"


def _ordered_unique(values: Any) -> list[str]:
    if values in (None, ""):
        return []
    if isinstance(values, str):
        candidates = [values]
    else:
        try:
            candidates = list(values)
        except TypeError:
            return []
    result = []
    seen = set()
    for item in candidates:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "no", "off"}:
            return False
        if normalized in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


__all__ = [
    "DetailSection",
    "DraftState",
    "ML_CLASS_SCORE_FIELDS",
    "MenuItem",
    "MORPHOLOGY_PRIMARY_BY_KEY",
    "MORPHOLOGY_PRIMARY_BY_VALUE",
    "MORPHOLOGY_PRIMARY_ITEMS",
    "MORPHOLOGY_SECONDARY_BY_KEY",
    "MORPHOLOGY_SECONDARY_BY_VALUE",
    "MORPHOLOGY_SECONDARY_ITEMS",
    "PHYSICAL_PRIMARY_BY_KEY",
    "PHYSICAL_PRIMARY_BY_VALUE",
    "PHYSICAL_PRIMARY_ITEMS",
    "PHYSICAL_SECONDARY_BY_KEY",
    "PHYSICAL_SECONDARY_BY_VALUE",
    "PHYSICAL_SECONDARY_ITEMS",
    "ReviewDraft",
    "compact_detail_lines",
    "detail_sections",
    "physical_primary_item_for_key",
    "physical_secondary_item_for_key",
    "physical_secondary_items_for",
    "primary_item_for_key",
    "secondary_item_for_key",
    "secondary_items_for",
]
