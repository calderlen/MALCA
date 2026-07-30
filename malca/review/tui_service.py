"""Persistence and queue-session support for the terminal reviewer.

The TUI deliberately owns only a small part of a review: primary morphology,
its (multi-select) secondary labels, confidence, and follow-up state.  Saves
therefore merge those values into the latest database row rather than writing
the draft as a complete review and accidentally erasing fields maintained by
the browser reviewer.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from malca.external_lc_manifest import (
    external_lc_manifest_path,
    normalize_external_lc_file_prefix,
    read_external_lc_manifest,
)
from malca.review.classification_labels import resolve_catalog_class
from malca.review.filter_schema import (
    is_dipper_contaminant_type_value,
    is_known_variable_type_value,
)
from malca.review.store import (
    count_queue,
    db_connect,
    ensure_review_db_schema,
    get_candidate_payload,
    get_review,
    load_app_state,
    query_queue,
    save_app_state,
    save_review as save_review_row,
)
from malca.review.taxonomy import label_for, selection_from_review
from malca.review.tui_photometry import (
    DEFAULT_TUI_EXTERNAL_PHOTOMETRY_SOURCES,
    TUI_EXTERNAL_PHOTOMETRY_AVAILABILITY_SOURCE_CHOICES,
    normalize_tui_external_photometry_availability_sources,
    normalize_tui_external_photometry_sources,
    tui_external_photometry_source_label,
)

if TYPE_CHECKING:
    from malca.review.tui_controller import ReviewDraft


_LAST_CANDIDATE_STATE_KEY = "review_last_candidate"
TUI_FILTER_STATE_KEY = "tui_queue_filter_state_v1"
CATALOG_TYPE_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("vsx", "VSX", "vsx_class"),
    ("gaia", "Gaia", "gaia_var_class"),
    ("asassn", "ASAS-SN", "asassn_var_type"),
    ("simbad", "SIMBAD", "simbad_otype"),
    ("ztf", "ZTF", "ztf_var_type"),
    ("microlens", "Microlensing", "microlens_catalog"),
    ("tns", "TNS", "tns_type"),
    ("alerce", "ALeRCE", "alerce_lc_class"),
    ("yso", "YSO", "yso_class"),
)
CATALOG_TYPE_EXCLUSION_FIELDS = {
    "vsx": "excluded_vsx_types",
    "gaia": "excluded_gaia_var_types",
    "asassn": "excluded_asassn_var_types",
    "simbad": "excluded_simbad_types",
    "ztf": "excluded_ztf_types",
    "microlens": "excluded_microlens_catalogs",
    "tns": "excluded_tns_types",
    "alerce": "excluded_alerce_classes",
    "yso": "excluded_yso_classes",
}


class EmptyQueueError(ValueError):
    """Raised when a filter change would replace the queue with no candidates."""


@dataclass(frozen=True)
class NumericRange:
    """Inclusive numeric range used by the dependency-light TUI filter model."""

    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        minimum = _optional_finite_float(self.minimum, name="minimum")
        maximum = _optional_finite_float(self.maximum, name="maximum")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("range minimum must not exceed maximum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @property
    def active(self) -> bool:
        return self.minimum is not None or self.maximum is not None

    def to_dict(self) -> dict[str, float | None]:
        return {"minimum": self.minimum, "maximum": self.maximum}

    @classmethod
    def from_value(cls, value: object) -> "NumericRange":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if isinstance(value, dict):
            return cls(value.get("minimum"), value.get("maximum"))
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return cls(value[0], value[1])
        raise ValueError(f"invalid numeric range: {value!r}")


_QUEUE_STATES = {"unreviewed", "reviewed", "followup", "all"}
_SIGNAL_LANES = {"all", "dip", "brightening", "mixed", "periodic"}
_THREE_WAY_MODES = {"any", "exclude", "only"}
_CATEGORICAL_LOGIC = {"all", "any"}
HIERARCHY_PREDICTION_CHOICES: dict[str, tuple[str, ...]] = {
    "predicted_hierarchy_gate": (
        "any",
        "artifact_or_nonvariable",
        "usable_astrophysical_variable",
    ),
    "predicted_primary_morphology": (
        "any",
        "dipper_dimming",
        "eb_geometric_periodic",
        "long_timescale_variable",
        "brightening_transient",
        "other_structured_variable",
    ),
    "predicted_hierarchical_class": (
        "any",
        "artifact_or_nonvariable",
        "dipper_dimming",
        "eb_geometric_periodic",
        "long_timescale_variable",
        "brightening_transient",
        "other_structured_variable",
    ),
    "predicted_quasi_periodic": (
        "any",
        "quasi_periodic",
        "not_quasi_periodic",
        "not_applicable",
    ),
    "predicted_microlensing_like": (
        "any",
        "microlensing_like",
        "not_microlensing_like",
        "not_applicable",
    ),
    "predicted_long_timescale_subtype": (
        "any",
        "long_period_variable",
        "long_term_variable",
        "not_applicable",
    ),
    "predicted_dipper_recurrence": (
        "any",
        "recurrent",
        "non_recurrent",
        "not_applicable",
    ),
}
_SORT_COLUMNS = {
    "candidate_id": "candidate_id",
    "prob_hierarchical_artifact_or_nonvariable": "prob_hierarchical_artifact_or_nonvariable",
    "prob_usable_astrophysical_variable": "prob_usable_astrophysical_variable",
    "prob_dipper_dimming": "prob_dipper_dimming",
    "prob_eb_geometric_periodic": "prob_eb_geometric_periodic",
    "prob_long_timescale_variable": "prob_long_timescale_variable",
    "prob_brightening_transient": "prob_brightening_transient",
    "prob_other_structured_variable": "prob_other_structured_variable",
    "prob_quasi_periodic_hierarchical": "prob_quasi_periodic_hierarchical",
    "prob_microlensing_hierarchical": "prob_microlensing_hierarchical",
    "prob_long_period_variable_hierarchical": "prob_long_period_variable_hierarchical",
    "prob_long_term_variable_hierarchical": "prob_long_term_variable_hierarchical",
    "prob_recurrent_dipper_hierarchical": "prob_recurrent_dipper_hierarchical",
    "prob_single_dipper_hierarchical": "prob_single_dipper_hierarchical",
    "prob_quasi_periodic_given_usable": "prob_quasi_periodic_given_usable",
    "prob_microlensing_given_brightening": "prob_microlensing_given_brightening",
    "prob_long_period_variable_given_long_timescale": "prob_long_period_variable_given_long_timescale",
    "prob_long_term_variable_given_long_timescale": "prob_long_term_variable_given_long_timescale",
    "prob_recurrent_given_dipper": "prob_recurrent_given_dipper",
    "prob_single_given_dipper": "prob_single_given_dipper",
    "dipper_score": "dipper_score",
    "jumper_score": "jumper_score",
    "q": "stats_variability_quasi_periodicity_q",
    "m": "stats_variability_flux_asymmetry_m",
    "g_magnitude": "phot_g_mean_mag",
    "period_days": "periodicity_period",
    "last_review": "updated_at",
    "confidence": "classification_confidence",
}
_RANGE_COLUMNS = {
    "prob_hierarchical_artifact_or_nonvariable": ("prob_hierarchical_artifact_or_nonvariable", "PRej"),
    "prob_usable_astrophysical_variable": ("prob_usable_astrophysical_variable", "PUse"),
    "prob_dipper_dimming": ("prob_dipper_dimming", "Pdip"),
    "prob_eb_geometric_periodic": ("prob_eb_geometric_periodic", "PEB"),
    "prob_long_timescale_variable": ("prob_long_timescale_variable", "PLong"),
    "prob_brightening_transient": ("prob_brightening_transient", "PBr"),
    "prob_other_structured_variable": ("prob_other_structured_variable", "POther"),
    "prob_quasi_periodic_hierarchical": ("prob_quasi_periodic_hierarchical", "PQP"),
    "prob_microlensing_hierarchical": ("prob_microlensing_hierarchical", "PML"),
    "prob_long_period_variable_hierarchical": ("prob_long_period_variable_hierarchical", "PLPV"),
    "prob_long_term_variable_hierarchical": ("prob_long_term_variable_hierarchical", "PLTV"),
    "prob_recurrent_dipper_hierarchical": ("prob_recurrent_dipper_hierarchical", "PRec"),
    "prob_single_dipper_hierarchical": ("prob_single_dipper_hierarchical", "PSingle"),
    "prob_quasi_periodic_given_usable": ("prob_quasi_periodic_given_usable", "PQP|U"),
    "prob_microlensing_given_brightening": ("prob_microlensing_given_brightening", "PML|Br"),
    "prob_long_period_variable_given_long_timescale": ("prob_long_period_variable_given_long_timescale", "PLPV|L"),
    "prob_long_term_variable_given_long_timescale": ("prob_long_term_variable_given_long_timescale", "PLTV|L"),
    "prob_recurrent_given_dipper": ("prob_recurrent_given_dipper", "PRec|D"),
    "prob_single_given_dipper": ("prob_single_given_dipper", "PSingle|D"),
    "dipper_score": ("dipper_score", "dip"),
    "jumper_score": ("jumper_score", "jump"),
    "q": ("stats_variability_quasi_periodicity_q", "Q"),
    "m": ("stats_variability_flux_asymmetry_m", "M"),
    "g_magnitude": ("phot_g_mean_mag", "G"),
    "period_days": ("periodicity_period", "P"),
    "confidence": ("classification_confidence", "conf"),
}
_SORT_LABELS = {
    "candidate_id": "ID",
    "prob_hierarchical_artifact_or_nonvariable": "PRej",
    "prob_usable_astrophysical_variable": "PUse",
    "prob_dipper_dimming": "Pdip",
    "prob_eb_geometric_periodic": "PEB",
    "prob_long_timescale_variable": "PLong",
    "prob_brightening_transient": "PBr",
    "prob_other_structured_variable": "POther",
    "prob_quasi_periodic_hierarchical": "PQP",
    "prob_microlensing_hierarchical": "PML",
    "prob_long_period_variable_hierarchical": "PLPV",
    "prob_long_term_variable_hierarchical": "PLTV",
    "prob_recurrent_dipper_hierarchical": "PRec",
    "prob_single_dipper_hierarchical": "PSingle",
    "prob_quasi_periodic_given_usable": "PQP|U",
    "prob_microlensing_given_brightening": "PML|Br",
    "prob_long_period_variable_given_long_timescale": "PLPV|L",
    "prob_long_term_variable_given_long_timescale": "PLTV|L",
    "prob_recurrent_given_dipper": "PRec|D",
    "prob_single_given_dipper": "PSingle|D",
    "dipper_score": "dip",
    "jumper_score": "jump",
    "q": "Q",
    "m": "M",
    "g_magnitude": "G",
    "period_days": "P",
    "last_review": "updated",
    "confidence": "conf",
}
_PREDICTION_SUMMARY_LABELS = {
    "predicted_hierarchy_gate": "MLgate",
    "predicted_primary_morphology": "MLprimary",
    "predicted_hierarchical_class": "MLclass",
    "predicted_quasi_periodic": "MLqp",
    "predicted_microlensing_like": "MLmicro",
    "predicted_long_timescale_subtype": "MLlong",
    "predicted_dipper_recurrence": "MLrec",
}


@dataclass(frozen=True)
class QueueFilterSpec:
    """Curated TUI filters translated directly to ``store.query_queue``.

    This deliberately exposes a small, stable filter vocabulary rather than
    mirroring every browser-sidebar field.  All matching is still performed by
    the canonical queue engine.
    """

    queue_state: str = "unreviewed"
    signal_lane: str = "all"
    show_external_lightcurves: bool = True
    external_lightcurve_sources: tuple[str, ...] = (
        DEFAULT_TUI_EXTERNAL_PHOTOMETRY_SOURCES
    )
    required_external_photometry_sources: tuple[str, ...] = ()
    excluded_external_photometry_sources: tuple[str, ...] = ()
    known_objects: str = "any"
    high_ruwe: str = "any"
    high_pm: str = "exclude"
    exclude_known_neighbors: bool = False
    exclude_dipper_contaminants: bool = False
    exclude_failed: bool = False
    neighbor_radius_arcsec: float = 15.0
    morphology_primary: tuple[str, ...] = ()
    physical_primary: tuple[str, ...] = ()
    excluded_vsx_types: tuple[str, ...] = ()
    excluded_gaia_var_types: tuple[str, ...] = ()
    excluded_asassn_var_types: tuple[str, ...] = ()
    excluded_simbad_types: tuple[str, ...] = ()
    excluded_ztf_types: tuple[str, ...] = ()
    excluded_microlens_catalogs: tuple[str, ...] = ()
    excluded_tns_types: tuple[str, ...] = ()
    excluded_alerce_classes: tuple[str, ...] = ()
    excluded_yso_classes: tuple[str, ...] = ()
    confidence: NumericRange = field(default_factory=NumericRange)
    prob_hierarchical_artifact_or_nonvariable: NumericRange = field(default_factory=NumericRange)
    prob_usable_astrophysical_variable: NumericRange = field(default_factory=NumericRange)
    prob_dipper_dimming: NumericRange = field(default_factory=NumericRange)
    prob_eb_geometric_periodic: NumericRange = field(default_factory=NumericRange)
    prob_long_timescale_variable: NumericRange = field(default_factory=NumericRange)
    prob_brightening_transient: NumericRange = field(default_factory=NumericRange)
    prob_other_structured_variable: NumericRange = field(default_factory=NumericRange)
    prob_quasi_periodic_hierarchical: NumericRange = field(default_factory=NumericRange)
    prob_microlensing_hierarchical: NumericRange = field(default_factory=NumericRange)
    prob_long_period_variable_hierarchical: NumericRange = field(default_factory=NumericRange)
    prob_long_term_variable_hierarchical: NumericRange = field(default_factory=NumericRange)
    prob_recurrent_dipper_hierarchical: NumericRange = field(default_factory=NumericRange)
    prob_single_dipper_hierarchical: NumericRange = field(default_factory=NumericRange)
    prob_quasi_periodic_given_usable: NumericRange = field(default_factory=NumericRange)
    prob_microlensing_given_brightening: NumericRange = field(default_factory=NumericRange)
    prob_long_period_variable_given_long_timescale: NumericRange = field(default_factory=NumericRange)
    prob_long_term_variable_given_long_timescale: NumericRange = field(default_factory=NumericRange)
    prob_recurrent_given_dipper: NumericRange = field(default_factory=NumericRange)
    prob_single_given_dipper: NumericRange = field(default_factory=NumericRange)
    predicted_hierarchy_gate: str = "any"
    predicted_primary_morphology: str = "any"
    predicted_hierarchical_class: str = "any"
    predicted_quasi_periodic: str = "any"
    predicted_microlensing_like: str = "any"
    predicted_long_timescale_subtype: str = "any"
    predicted_dipper_recurrence: str = "any"
    dipper_score: NumericRange = field(default_factory=NumericRange)
    jumper_score: NumericRange = field(default_factory=NumericRange)
    q: NumericRange = field(default_factory=NumericRange)
    m: NumericRange = field(default_factory=NumericRange)
    g_magnitude: NumericRange = field(default_factory=NumericRange)
    period_days: NumericRange = field(default_factory=NumericRange)
    sort_by: str = "candidate_id"
    sort_desc: bool = False
    categorical_logic: str = "all"

    def __post_init__(self) -> None:
        queue_state = _normalized_choice(
            self.queue_state,
            _QUEUE_STATES,
            name="queue state",
            aliases={"follow-up": "followup", "follow_up": "followup"},
        )
        signal_lane = _normalized_choice(
            self.signal_lane,
            _SIGNAL_LANES,
            name="signal lane",
            aliases={
                "dip-only": "dip",
                "dip_only": "dip",
                "brightening-only": "brightening",
                "brightening_only": "brightening",
            },
        )
        known_objects = _normalized_choice(
            self.known_objects, _THREE_WAY_MODES, name="known-object mode"
        )
        high_ruwe = _normalized_choice(
            self.high_ruwe, _THREE_WAY_MODES, name="high-RUWE mode"
        )
        high_pm = _normalized_choice(
            self.high_pm, _THREE_WAY_MODES, name="high-PM mode"
        )
        categorical_logic = _normalized_choice(
            self.categorical_logic,
            _CATEGORICAL_LOGIC,
            name="categorical logic",
            aliases={"and": "all", "or": "any"},
        )
        prediction_choices = {
            field_name: _normalized_choice(
                getattr(self, field_name),
                set(choices),
                name=field_name.replace("_", " "),
            )
            for field_name, choices in HIERARCHY_PREDICTION_CHOICES.items()
        }
        sort_by = str(self.sort_by or "candidate_id").strip().lower()
        sort_by = {
            "pdip": "prob_dipper_dimming",
            "prob_dipper_like": "prob_dipper_dimming",
            "prob_dipper": "prob_dipper_dimming",
            "prob_artifact_or_bad_photometry": "prob_hierarchical_artifact_or_nonvariable",
            "prob_nonvariable_or_low_snr": "prob_hierarchical_artifact_or_nonvariable",
            "prob_artifact_or_nonvariable": "prob_hierarchical_artifact_or_nonvariable",
            "prob_brightening_event": "prob_brightening_transient",
            "prob_eclipsing_binary_like": "prob_eb_geometric_periodic",
            "prob_long_period_variable": "prob_long_period_variable_hierarchical",
            "prob_long_term_variable": "prob_long_term_variable_hierarchical",
            "prob_microlensing": "prob_microlensing_hierarchical",
            "prob_quasi_periodic": "prob_quasi_periodic_hierarchical",
            "g": "g_magnitude",
            "period": "period_days",
            "updated_at": "last_review",
        }.get(sort_by, sort_by)
        if sort_by not in _SORT_COLUMNS:
            choices = ", ".join(sorted(_SORT_COLUMNS))
            raise ValueError(f"invalid sort column {self.sort_by!r}; choose one of {choices}")

        radius = _optional_finite_float(
            self.neighbor_radius_arcsec, name="catalog-neighbor radius"
        )
        if radius is None or radius < 0:
            raise ValueError("catalog-neighbor radius must be a non-negative number")

        object.__setattr__(self, "queue_state", queue_state)
        object.__setattr__(self, "signal_lane", signal_lane)
        object.__setattr__(
            self,
            "show_external_lightcurves",
            bool(self.show_external_lightcurves),
        )
        object.__setattr__(
            self,
            "external_lightcurve_sources",
            normalize_tui_external_photometry_sources(
                self.external_lightcurve_sources
            ),
        )
        required_external = normalize_tui_external_photometry_availability_sources(
            self.required_external_photometry_sources
        )
        excluded_external = normalize_tui_external_photometry_availability_sources(
            self.excluded_external_photometry_sources
        )
        overlap = set(required_external) & set(excluded_external)
        if overlap:
            labels = ", ".join(
                tui_external_photometry_source_label(source)
                for source in required_external
                if source in overlap
            )
            raise ValueError(
                f"external photometry cannot be both required and absent: {labels}"
            )
        object.__setattr__(
            self,
            "required_external_photometry_sources",
            required_external,
        )
        object.__setattr__(
            self,
            "excluded_external_photometry_sources",
            excluded_external,
        )
        object.__setattr__(self, "known_objects", known_objects)
        object.__setattr__(self, "high_ruwe", high_ruwe)
        object.__setattr__(self, "high_pm", high_pm)
        object.__setattr__(self, "categorical_logic", categorical_logic)
        object.__setattr__(self, "sort_by", sort_by)
        object.__setattr__(self, "neighbor_radius_arcsec", radius)
        for field_name, value in prediction_choices.items():
            object.__setattr__(self, field_name, value)
        object.__setattr__(
            self, "morphology_primary", _normalized_values(self.morphology_primary)
        )
        object.__setattr__(
            self, "physical_primary", _normalized_values(self.physical_primary)
        )
        for field_name in CATALOG_TYPE_EXCLUSION_FIELDS.values():
            object.__setattr__(
                self,
                field_name,
                _normalized_values(getattr(self, field_name)),
            )
        for field_name in _RANGE_COLUMNS:
            object.__setattr__(
                self, field_name, NumericRange.from_value(getattr(self, field_name))
            )

    @classmethod
    def default(cls, *, only_unreviewed: bool = True) -> "QueueFilterSpec":
        return cls(queue_state="unreviewed" if only_unreviewed else "all")

    def to_query_filters(self) -> dict[str, object]:
        """Return canonical filter parameters accepted by ``query_queue``."""

        filters: dict[str, object] = {
            "select_filter_mode": "include",
            "select_filter_logic": (
                "and" if self.categorical_logic == "all" else "or"
            ),
            "sort_cols": [_SORT_COLUMNS[self.sort_by]],
            "sort_desc": bool(self.sort_desc),
        }

        if self.queue_state == "unreviewed":
            filters["only_unreviewed"] = True
        elif self.queue_state in {"reviewed", "followup"}:
            filters["workflow_status_exact"] = (
                "reviewed" if self.queue_state == "reviewed" else "needs_followup"
            )

        if self.signal_lane == "dip":
            filters.update(dip_significant_mode="True", jump_significant_mode="False")
        elif self.signal_lane == "brightening":
            filters.update(dip_significant_mode="False", jump_significant_mode="True")
        elif self.signal_lane == "mixed":
            filters.update(dip_significant_mode="True", jump_significant_mode="True")
        elif self.signal_lane == "periodic":
            filters["periodic_flag_mode"] = "True"

        if self.known_objects != "any":
            filters["vetting_likely_known_mode"] = (
                "False" if self.known_objects == "exclude" else "True"
            )
        if self.high_ruwe != "any":
            filters["high_ruwe_flag_mode"] = (
                "False" if self.high_ruwe == "exclude" else "True"
            )
        if self.high_pm != "any":
            filters["high_pm_flag_mode"] = (
                "False" if self.high_pm == "exclude" else "True"
            )
        if self.exclude_known_neighbors:
            filters["exclude_known_catalog_neighbors"] = True
        if self.exclude_dipper_contaminants:
            filters["exclude_dipper_catalog_neighbors"] = True
        if self.exclude_known_neighbors or self.exclude_dipper_contaminants:
            filters["catalog_neighbor_radius_arcsec"] = self.neighbor_radius_arcsec
        if self.exclude_failed:
            filters["require_failed_any_false"] = True

        if self.morphology_primary:
            filters["exclude_morphology_primary"] = list(self.morphology_primary)
        if self.physical_primary:
            filters["exclude_physical_primary"] = list(self.physical_primary)
        for field_name in HIERARCHY_PREDICTION_CHOICES:
            value = getattr(self, field_name)
            if value != "any":
                # The canonical queue engine uses ``exclude_*`` as its
                # multi-select payload name; include mode above makes these
                # exact predicted-class inclusions.
                filters[f"exclude_{field_name}"] = [value]

        catalog_exclusions = {
            column: list(getattr(self, CATALOG_TYPE_EXCLUSION_FIELDS[catalog]))
            for catalog, _label, column in CATALOG_TYPE_SOURCES
            if getattr(self, CATALOG_TYPE_EXCLUSION_FIELDS[catalog])
        }
        if catalog_exclusions:
            filters["catalog_type_exclusions"] = catalog_exclusions

        for field_name, (column, _label) in _RANGE_COLUMNS.items():
            value_range = getattr(self, field_name)
            if value_range.minimum is not None:
                filters[f"min_{column}"] = value_range.minimum
            if value_range.maximum is not None:
                filters[f"max_{column}"] = value_range.maximum
        return filters

    def summary_parts(self) -> tuple[str, ...]:
        """Return compact, human-readable parts for the main TUI header."""

        parts = [
            {
                "unreviewed": "unreviewed",
                "reviewed": "reviewed",
                "followup": "follow-up",
                "all": "all",
            }[self.queue_state]
        ]
        if self.signal_lane != "all":
            parts.append(
                {
                    "dip": "dip",
                    "brightening": "brightening",
                    "mixed": "mixed",
                    "periodic": "periodic",
                }[self.signal_lane]
            )
        if self.known_objects != "any":
            parts.append("known-" if self.known_objects == "exclude" else "known-only")
        if self.high_ruwe != "any":
            parts.append("RUWE-" if self.high_ruwe == "exclude" else "high-RUWE")
        if self.high_pm != "any":
            parts.append("PM-" if self.high_pm == "exclude" else "high-PM")
        if self.exclude_known_neighbors:
            parts.append("known-neighbor-")
        if self.exclude_dipper_contaminants:
            parts.append("contaminant-")
        if self.exclude_failed:
            parts.append("failed-")
        for source in self.required_external_photometry_sources:
            parts.append(
                f"phot+{tui_external_photometry_source_label(source)}"
            )
        for source in self.excluded_external_photometry_sources:
            parts.append(
                f"phot−{tui_external_photometry_source_label(source)}"
            )
        for catalog, label, _column in CATALOG_TYPE_SOURCES:
            excluded = getattr(self, CATALOG_TYPE_EXCLUSION_FIELDS[catalog])
            if excluded:
                parts.append(f"{label}\N{MINUS SIGN}{len(excluded)}")
        if self.morphology_primary:
            parts.append(_selection_summary("morph", self.morphology_primary))
        if self.physical_primary:
            parts.append(_selection_summary("phys", self.physical_primary))
        active_predictions = [
            (
                _PREDICTION_SUMMARY_LABELS[field_name],
                getattr(self, field_name),
            )
            for field_name in HIERARCHY_PREDICTION_CHOICES
            if getattr(self, field_name) != "any"
        ]
        for label, value in active_predictions:
            parts.append(f"{label}:{value}")
        if (
            (
                self.morphology_primary
                or self.physical_primary
                or active_predictions
            )
            and self.categorical_logic == "any"
        ):
            parts.append("ANY")
        for field_name, (_column, label) in _RANGE_COLUMNS.items():
            value_range = getattr(self, field_name)
            if value_range.active:
                parts.append(_range_summary(label, value_range))
        arrow = "\N{DOWNWARDS ARROW}" if self.sort_desc else "\N{UPWARDS ARROW}"
        parts.append(f"{_SORT_LABELS[self.sort_by]}{arrow}")
        return tuple(parts)

    def summary(self) -> str:
        return " \N{MIDDLE DOT} ".join(self.summary_parts())

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "queue_state": self.queue_state,
            "signal_lane": self.signal_lane,
            "show_external_lightcurves": self.show_external_lightcurves,
            "external_lightcurve_sources": list(self.external_lightcurve_sources),
            "required_external_photometry_sources": list(
                self.required_external_photometry_sources
            ),
            "excluded_external_photometry_sources": list(
                self.excluded_external_photometry_sources
            ),
            "known_objects": self.known_objects,
            "high_ruwe": self.high_ruwe,
            "high_pm": self.high_pm,
            "exclude_known_neighbors": self.exclude_known_neighbors,
            "exclude_dipper_contaminants": self.exclude_dipper_contaminants,
            "exclude_failed": self.exclude_failed,
            "neighbor_radius_arcsec": self.neighbor_radius_arcsec,
            "morphology_primary": list(self.morphology_primary),
            "physical_primary": list(self.physical_primary),
            "excluded_vsx_types": list(self.excluded_vsx_types),
            "excluded_gaia_var_types": list(self.excluded_gaia_var_types),
            "excluded_asassn_var_types": list(self.excluded_asassn_var_types),
            "excluded_simbad_types": list(self.excluded_simbad_types),
            "excluded_ztf_types": list(self.excluded_ztf_types),
            "excluded_microlens_catalogs": list(self.excluded_microlens_catalogs),
            "excluded_tns_types": list(self.excluded_tns_types),
            "excluded_alerce_classes": list(self.excluded_alerce_classes),
            "excluded_yso_classes": list(self.excluded_yso_classes),
            "sort_by": self.sort_by,
            "sort_desc": self.sort_desc,
            "categorical_logic": self.categorical_logic,
        }
        for field_name in HIERARCHY_PREDICTION_CHOICES:
            data[field_name] = getattr(self, field_name)
        for field_name in _RANGE_COLUMNS:
            data[field_name] = getattr(self, field_name).to_dict()
        return data

    @classmethod
    def from_dict(cls, value: dict[str, object] | None) -> "QueueFilterSpec":
        raw = dict(value or {})
        # Migrate persisted flat-model ranges into the closest hierarchical
        # score so old saved TUI state does not prevent launch.
        legacy_ranges = {
            "prob_dipper_like": "prob_dipper_dimming",
            "prob_dipper": "prob_dipper_dimming",
            "prob_artifact_or_bad_photometry": "prob_hierarchical_artifact_or_nonvariable",
            "prob_nonvariable_or_low_snr": "prob_hierarchical_artifact_or_nonvariable",
            "prob_artifact_or_nonvariable": "prob_hierarchical_artifact_or_nonvariable",
            "prob_brightening_event": "prob_brightening_transient",
            "prob_eclipsing_binary_like": "prob_eb_geometric_periodic",
            "prob_long_period_variable": "prob_long_period_variable_hierarchical",
            "prob_long_term_variable": "prob_long_term_variable_hierarchical",
            "prob_microlensing": "prob_microlensing_hierarchical",
            "prob_quasi_periodic": "prob_quasi_periodic_hierarchical",
        }
        for legacy, replacement in legacy_ranges.items():
            if replacement not in raw and legacy in raw:
                raw[replacement] = raw[legacy]
            raw.pop(legacy, None)
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: raw[key] for key in raw.keys() & allowed})


@dataclass(frozen=True)
class CatalogTypeStat:
    """Campaign-local count and interpretation for one catalog class."""

    catalog: str
    catalog_label: str
    column: str
    value: str
    count: int
    total_candidates: int
    description: str
    known_variable: bool
    dipper_contaminant: bool
    uncertain: bool

    @property
    def fraction(self) -> float:
        if self.total_candidates <= 0:
            return 0.0
        return float(self.count) / float(self.total_candidates)


@dataclass(frozen=True)
class CandidateRecord:
    """Candidate data needed by the TUI and its lightweight renderer."""

    candidate_id: str
    asas_sn_id: str | None
    lc_path: Path | None
    source_path: str | None
    payload: dict[str, Any]


def _external_results_root_for_review_db(
    db_path: Path,
    explicit_root: str | Path | None = None,
) -> Path | None:
    candidates: list[Path] = []
    if explicit_root is not None:
        explicit = Path(explicit_root).expanduser()
        candidates.append(explicit if explicit.name == "results" else explicit / "results")
    candidates.extend(
        (
            db_path.parent.parent / "results",
            db_path.parent / "results",
        )
    )
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        manifest_path = external_lc_manifest_path(candidate)
        if manifest_path is not None and manifest_path.is_file():
            return candidate
    return None


def _external_photometry_candidate_ids(
    results_root: Path | None,
) -> dict[str, frozenset[str]]:
    by_source = {
        source: set()
        for source in TUI_EXTERNAL_PHOTOMETRY_AVAILABILITY_SOURCE_CHOICES
    }
    if results_root is None:
        return {source: frozenset() for source in by_source}
    manifest = read_external_lc_manifest(results_root)
    if manifest.empty:
        return {source: frozenset() for source in by_source}
    for raw_source, raw_candidate_id in zip(
        manifest["file_prefix"],
        manifest["candidate_id"],
    ):
        source = normalize_external_lc_file_prefix(raw_source)
        candidate_id = str(raw_candidate_id or "").strip()
        if source in by_source and candidate_id:
            by_source[source].add(candidate_id)
    return {
        source: frozenset(candidate_ids)
        for source, candidate_ids in by_source.items()
    }


class ReviewRepository:
    """A short-lived, synchronous review session over one SQLite database.

    ``candidate_ids`` is a snapshot taken at construction time.  It does not
    shrink as reviews are saved, which makes next/previous navigation stable
    for a terminal session started with ``only_unreviewed=True``.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        only_unreviewed: bool = True,
        candidate_query: str | None = None,
        reviewer: str = "calder",
        filter_spec: QueueFilterSpec | None = None,
        restore_filter_state: bool = True,
        external_results_root: str | Path | None = None,
    ) -> None:
        self.db_path = Path(db_path).expanduser()
        self.reviewer = str(reviewer)
        self.external_results_root = _external_results_root_for_review_db(
            self.db_path,
            external_results_root,
        )
        self._external_photometry_ids = _external_photometry_candidate_ids(
            self.external_results_root
        )

        ensure_review_db_schema(self.db_path)
        with closing(db_connect(self.db_path)) as conn:
            active_filter = filter_spec
            # ``--all`` remains an explicit one-shot override.  Otherwise a
            # normal unreviewed startup restores the last TUI-only filter set.
            if (
                active_filter is None
                and restore_filter_state
                and bool(only_unreviewed)
            ):
                active_filter = self._load_filter_spec(conn)
            if active_filter is None:
                active_filter = QueueFilterSpec.default(
                    only_unreviewed=bool(only_unreviewed)
                )

            queue = query_queue(
                conn,
                filters=self._query_filters(active_filter),
                ids_only=True,
            )
            snapshot = tuple(str(value) for value in queue["candidate_id"].tolist())
            self._filtered_candidate_ids = snapshot
            self.filter_spec = active_filter
            self.only_unreviewed = active_filter.queue_state == "unreviewed"
            self.search_override = False

            explicit_query = str(candidate_query or "").strip()
            if explicit_query:
                candidate_id = self._resolve_candidate_query(conn, explicit_query)
                if candidate_id in snapshot:
                    self.candidate_ids = snapshot
                    self.start_index = snapshot.index(candidate_id)
                else:
                    # A reviewed candidate requested from an unreviewed queue is
                    # still useful for inspection/editing, but should not pull in
                    # the rest of the inactive queue.
                    self.candidate_ids = (candidate_id,)
                    self.start_index = 0
                    self.search_override = True
                return

            self.candidate_ids = snapshot
            saved_candidate = str(
                load_app_state(conn, _LAST_CANDIDATE_STATE_KEY, "") or ""
            ).strip()
            self.start_index = (
                snapshot.index(saved_candidate) if saved_candidate in snapshot else 0
            )

    @staticmethod
    def _load_filter_spec(conn: Any) -> QueueFilterSpec | None:
        raw = str(load_app_state(conn, TUI_FILTER_STATE_KEY, "") or "").strip()
        if not raw:
            return None
        try:
            decoded = json.loads(raw)
            if not isinstance(decoded, dict):
                return None
            values = decoded.get("filters", decoded)
            if not isinstance(values, dict):
                return None
            return QueueFilterSpec.from_dict(values)
        except (TypeError, ValueError, json.JSONDecodeError):
            # A stale/corrupt UI preference must never prevent review startup.
            return None

    def load_filter_spec(self) -> QueueFilterSpec | None:
        """Load the separately persisted TUI filters, if they are valid."""

        with closing(db_connect(self.db_path)) as conn:
            return self._load_filter_spec(conn)

    def persist_filter_spec(self, spec: QueueFilterSpec | None = None) -> None:
        """Persist filters under a key that is independent of the Dash UI."""

        selected = spec if spec is not None else self.filter_spec
        payload = json.dumps(
            {"version": 1, "filters": selected.to_dict()},
            sort_keys=True,
            separators=(",", ":"),
        )
        with closing(db_connect(self.db_path)) as conn:
            save_app_state(conn, TUI_FILTER_STATE_KEY, payload)

    def preview_filter_count(self, spec: QueueFilterSpec) -> int:
        """Return a prospective match count without changing session state."""

        with closing(db_connect(self.db_path)) as conn:
            return count_queue(conn, filters=self._query_filters(spec))

    def external_photometry_counts(self) -> dict[str, int]:
        """Return manifest-backed availability counts for the active campaign."""

        return {
            source: len(candidate_ids)
            for source, candidate_ids in self._external_photometry_ids.items()
        }

    def _query_filters(self, spec: QueueFilterSpec) -> dict[str, object]:
        filters = spec.to_query_filters()
        required_sources = spec.required_external_photometry_sources
        excluded_sources = spec.excluded_external_photometry_sources
        if not required_sources and not excluded_sources:
            return filters

        required_ids: set[str] | None = None
        for source in required_sources:
            source_ids = set(self._external_photometry_ids.get(source, ()))
            required_ids = (
                source_ids
                if required_ids is None
                else required_ids & source_ids
            )
        excluded_ids: set[str] = set()
        for source in excluded_sources:
            excluded_ids.update(self._external_photometry_ids.get(source, ()))

        membership: dict[str, object] = {}
        if required_ids is not None:
            membership["required"] = sorted(required_ids)
        if excluded_ids:
            membership["excluded"] = sorted(excluded_ids)
        filters["candidate_id_membership"] = membership
        return filters

    def catalog_type_stats(self) -> tuple[CatalogTypeStat, ...]:
        """Return all direct catalog types stored in this campaign database."""

        stats: list[CatalogTypeStat] = []
        with closing(db_connect(self.db_path)) as conn:
            total_row = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()
            total_candidates = int(total_row[0] if total_row is not None else 0)
            actual_columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(candidates)").fetchall()
            }
            for catalog, catalog_label, column in CATALOG_TYPE_SOURCES:
                if column not in actual_columns:
                    continue
                rows = conn.execute(
                    f"""
                    SELECT TRIM({column}) AS catalog_type, COUNT(*) AS n
                    FROM candidates
                    WHERE {column} IS NOT NULL
                      AND TRIM({column}) <> ''
                    GROUP BY TRIM({column})
                    ORDER BY n DESC, catalog_type COLLATE NOCASE
                    """
                ).fetchall()
                for raw_value, raw_count in rows:
                    value = str(raw_value or "").strip()
                    if not value:
                        continue
                    resolution = resolve_catalog_class(column, value)
                    description = "; ".join(resolution.descriptions)
                    if not description:
                        description = "No catalog-class description is available."
                    stats.append(
                        CatalogTypeStat(
                            catalog=catalog,
                            catalog_label=catalog_label,
                            column=column,
                            value=value,
                            count=int(raw_count),
                            total_candidates=total_candidates,
                            description=description,
                            known_variable=is_known_variable_type_value(column, value),
                            dipper_contaminant=is_dipper_contaminant_type_value(
                                column,
                                value,
                            ),
                            uncertain=bool(resolution.uncertain),
                        )
                    )
        return tuple(stats)

    def apply_filters(
        self,
        spec: QueueFilterSpec,
        *,
        anchor_candidate_id: str | None = None,
        persist: bool = True,
    ) -> int:
        """Replace the navigation snapshot after a successful non-empty query.

        The current candidate is retained when it still matches.  Query,
        validation, empty-result, and persistence failures all occur before the
        existing tuple or index is mutated.
        """

        query_filters = self._query_filters(spec)
        with closing(db_connect(self.db_path)) as conn:
            queue = query_queue(conn, filters=query_filters, ids_only=True)
            snapshot = tuple(str(value) for value in queue["candidate_id"].tolist())
            if not snapshot:
                raise EmptyQueueError(
                    "No candidates match these filters; the current queue was kept."
                )
            if persist:
                payload = json.dumps(
                    {"version": 1, "filters": spec.to_dict()},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                save_app_state(conn, TUI_FILTER_STATE_KEY, payload)

        anchor = str(anchor_candidate_id or "").strip()
        self.filter_spec = spec
        self.only_unreviewed = spec.queue_state == "unreviewed"
        self._filtered_candidate_ids = snapshot
        self.candidate_ids = snapshot
        self.start_index = snapshot.index(anchor) if anchor in snapshot else 0
        self.search_override = False
        return self.start_index

    def search_candidate(self, query: str) -> tuple[int, bool]:
        """Jump to an exact candidate or ASAS-SN identifier.

        Candidates already present in the active snapshot keep that snapshot.
        A global match outside it becomes a one-candidate inspection snapshot;
        applying ``F`` filters again returns to a canonical filtered queue.
        The returned boolean reports whether that global override was needed.
        """

        text = str(query or "").strip()
        if not text:
            raise ValueError("Enter a candidate or ASAS-SN ID")
        with closing(db_connect(self.db_path)) as conn:
            candidate_id = self._resolve_candidate_query(conn, text)
        filtered_snapshot = self._filtered_candidate_ids
        if candidate_id in filtered_snapshot:
            self.candidate_ids = filtered_snapshot
            index = filtered_snapshot.index(candidate_id)
            self.start_index = index
            self.search_override = False
            return index, False
        self.candidate_ids = (candidate_id,)
        self.start_index = 0
        self.search_override = True
        return 0, True

    @staticmethod
    def _resolve_candidate_query(conn: Any, query: str) -> str:
        """Resolve an exact candidate ID first, then an exact ASAS-SN ID."""

        row = conn.execute(
            "SELECT candidate_id FROM candidates WHERE candidate_id = ?",
            (query,),
        ).fetchone()
        if row is not None:
            return str(row[0])

        rows = conn.execute(
            """
            SELECT candidate_id
            FROM candidates
            WHERE asas_sn_id = ?
            ORDER BY candidate_id
            """,
            (query,),
        ).fetchall()
        if not rows:
            raise ValueError(f"No candidate matches {query!r}")
        if len(rows) > 1:
            matches = ", ".join(str(match[0]) for match in rows[:5])
            suffix = " ..." if len(rows) > 5 else ""
            raise ValueError(
                f"ASAS-SN ID {query!r} matches multiple candidates: {matches}{suffix}"
            )
        return str(rows[0][0])

    def load(self, candidate_id: str) -> tuple[CandidateRecord, dict[str, Any]]:
        """Load the canonical merged payload and current review for a candidate."""

        candidate_id = str(candidate_id)
        with closing(db_connect(self.db_path)) as conn:
            row = conn.execute(
                """
                SELECT candidate_id, asas_sn_id, lc_path, source_path
                FROM candidates
                WHERE candidate_id = ?
                """,
                (candidate_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown candidate_id: {candidate_id}")

            payload = get_candidate_payload(conn, candidate_id)
            review = get_review(conn, candidate_id)

        asas_sn_id = _optional_text(row[1]) or _optional_text(payload.get("asas_sn_id"))
        lc_path_text = _optional_text(row[2]) or _optional_text(payload.get("lc_path"))
        source_path = _optional_text(row[3]) or _optional_text(payload.get("source_path"))
        record = CandidateRecord(
            candidate_id=str(row[0]),
            asas_sn_id=asas_sn_id,
            lc_path=Path(lc_path_text).expanduser() if lc_path_text else None,
            source_path=source_path,
            payload=payload,
        )
        return record, review

    def save(
        self,
        candidate_id: str,
        draft: ReviewDraft,
        increment_pass: bool,
        event_type: str,
    ) -> dict[str, Any]:
        """Merge a TUI draft into the latest review row and commit it."""

        candidate_id = str(candidate_id)
        with closing(db_connect(self.db_path)) as conn:
            exists = conn.execute(
                "SELECT 1 FROM candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if exists is None:
                raise KeyError(f"Unknown candidate_id: {candidate_id}")

            current = get_review(conn, candidate_id)
            selection = selection_from_review(current)
            current_pass = max(1, int(current.get("review_pass") or 1))
            review_pass = current_pass + 1 if increment_pass else current_pass
            secondaries = list(draft.morphology_secondaries or ())
            if hasattr(draft, "physical_secondary"):
                physical_secondary = draft.physical_secondary
            elif draft.physical_primary == selection.get("physical_primary"):
                physical_secondary = selection.get("physical_secondary")
            else:
                physical_secondary = None
            notes = (
                str(draft.notes)
                if hasattr(draft, "notes")
                else str(current.get("notes") or "")
            )
            workflow_status = (
                "needs_followup" if bool(draft.needs_followup) else "reviewed"
            )

            save_review_row(
                conn,
                candidate_id=candidate_id,
                event_class=str(current.get("event_class") or "unclassified"),
                review_pass=review_pass,
                notes=notes,
                workflow_status=workflow_status,
                disposition=selection.get("disposition") or "keep",
                morphology_primary=draft.morphology_primary,
                # Both representations are supplied so an empty TUI selection
                # explicitly clears previously saved secondary labels.
                morphology_secondary=secondaries[0] if secondaries else None,
                morphology_secondary_json=secondaries,
                morphology_polarity=selection.get("morphology_polarity"),
                morphology_recurrence=selection.get("morphology_recurrence"),
                baseline_behavior=selection.get("baseline_behavior"),
                physical_primary=draft.physical_primary,
                physical_secondary=physical_secondary,
                classification_confidence=draft.confidence,
                priority_tags=selection.get("priority_tags"),
                evidence_flags=selection.get("evidence_flags"),
                model_tags=selection.get("model_tags"),
                duplicate_of=selection.get("duplicate_of"),
                known_object_id=selection.get("known_object_id"),
                known_object_source=selection.get("known_object_source"),
                legacy_review_json=str(current.get("legacy_review_json") or "{}"),
                reviewer=self.reviewer,
                event_type=str(event_type),
            )
            # save_review_row commits before returning.  Re-read on the same
            # connection so callers receive the normalized, persisted form.
            return get_review(conn, candidate_id)

    def persist_last_candidate(self, candidate_id: str) -> None:
        """Persist navigation position for the next non-explicit session."""

        candidate_id = str(candidate_id).strip()
        if not candidate_id:
            raise ValueError("candidate_id must not be empty")
        with closing(db_connect(self.db_path)) as conn:
            save_app_state(conn, _LAST_CANDIDATE_STATE_KEY, candidate_id)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_finite_float(value: object, *, name: str) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _normalized_choice(
    value: object,
    choices: set[str],
    *,
    name: str,
    aliases: dict[str, str] | None = None,
) -> str:
    normalized = str(value or "").strip().lower()
    normalized = (aliases or {}).get(normalized, normalized)
    if normalized not in choices:
        rendered = ", ".join(sorted(choices))
        raise ValueError(f"invalid {name} {value!r}; choose one of {rendered}")
    return normalized


def _normalized_values(values: object) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raw_values = [values]
    else:
        raw_values = list(values)  # type: ignore[arg-type]
    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        text = str(value or "").strip()
        if text and text not in seen:
            normalized.append(text)
            seen.add(text)
    return tuple(normalized)


def _format_number(value: float) -> str:
    return f"{value:g}"


def _range_summary(label: str, value_range: NumericRange) -> str:
    minimum = value_range.minimum
    maximum = value_range.maximum
    if minimum is not None and maximum is not None:
        if minimum == maximum:
            return f"{label}={_format_number(minimum)}"
        return f"{label}[{_format_number(minimum)},{_format_number(maximum)}]"
    if minimum is not None:
        return f"{label}\N{GREATER-THAN OR EQUAL TO}{_format_number(minimum)}"
    assert maximum is not None
    return f"{label}\N{LESS-THAN OR EQUAL TO}{_format_number(maximum)}"


def _selection_summary(prefix: str, values: tuple[str, ...]) -> str:
    labels = [label_for(value) for value in values]
    if len(labels) == 1:
        return f"{prefix}:{labels[0]}"
    return f"{prefix}:{labels[0]}+{len(labels) - 1}"
