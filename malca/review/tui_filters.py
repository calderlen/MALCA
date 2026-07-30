"""Pure filter-overlay state for the MALCA terminal reviewer.

The curses shell only needs to draw rows and forward left/right/toggle actions.
Keeping the editor here makes those actions deterministic and testable without
starting a terminal or opening the review database.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable

from malca.review.taxonomy import label_for
from malca.review.tui_service import (
    CATALOG_TYPE_EXCLUSION_FIELDS,
    CATALOG_TYPE_SOURCES,
    HIERARCHY_PREDICTION_CHOICES,
    NumericRange,
    QueueFilterSpec,
)
from malca.review.tui_photometry import (
    TUI_EXTERNAL_PHOTOMETRY_AVAILABILITY_SOURCE_CHOICES,
    TUI_EXTERNAL_PHOTOMETRY_SOURCE_CHOICES,
    tui_external_photometry_source_label,
)


@dataclass(frozen=True)
class FilterRow:
    key: str
    label: str
    value: str
    kind: str = "choice"


_CHOICES: dict[str, tuple[Any, ...]] = {
    "queue_state": ("unreviewed", "reviewed", "followup", "all"),
    "signal_lane": ("all", "dip", "brightening", "mixed", "periodic"),
    "show_external_lightcurves": (True, False),
    "known_objects": ("any", "exclude", "only"),
    "high_ruwe": ("any", "exclude", "only"),
    "high_pm": ("exclude", "any", "only"),
    "exclude_known_neighbors": (False, True),
    "exclude_dipper_contaminants": (False, True),
    "exclude_failed": (False, True),
    "categorical_logic": ("all", "any"),
    **HIERARCHY_PREDICTION_CHOICES,
    "sort_by": (
        "candidate_id",
        "prob_hierarchical_artifact_or_nonvariable",
        "prob_usable_astrophysical_variable",
        "prob_dipper_dimming",
        "prob_eb_geometric_periodic",
        "prob_long_timescale_variable",
        "prob_brightening_transient",
        "prob_other_structured_variable",
        "prob_quasi_periodic_hierarchical",
        "prob_microlensing_hierarchical",
        "prob_long_period_variable_hierarchical",
        "prob_long_term_variable_hierarchical",
        "prob_recurrent_dipper_hierarchical",
        "prob_single_dipper_hierarchical",
        "prob_quasi_periodic_given_usable",
        "prob_microlensing_given_brightening",
        "prob_long_period_variable_given_long_timescale",
        "prob_long_term_variable_given_long_timescale",
        "prob_recurrent_given_dipper",
        "prob_single_given_dipper",
        "dipper_score",
        "jumper_score",
        "q",
        "m",
        "g_magnitude",
        "period_days",
        "last_review",
        "confidence",
    ),
    "sort_desc": (False, True),
}

_EXTERNAL_PHOTOMETRY_ROWS: tuple[tuple[str, str, str], ...] = tuple(
    (
        f"external_source_{source}",
        tui_external_photometry_source_label(source),
        "external_source",
    )
    for source in TUI_EXTERNAL_PHOTOMETRY_SOURCE_CHOICES
)
_EXTERNAL_PHOTOMETRY_AVAILABILITY_ROWS: tuple[
    tuple[str, str, str], ...
] = tuple(
    (
        f"external_availability_{source}",
        tui_external_photometry_source_label(source),
        "external_availability",
    )
    for source in TUI_EXTERNAL_PHOTOMETRY_AVAILABILITY_SOURCE_CHOICES
)
_EXTERNAL_AVAILABILITY_MODES = ("any", "required", "absent")

_PROBABILITY_RANGE_PRESETS: tuple[tuple[str, NumericRange], ...] = (
    ("unrestricted", NumericRange()),
    (">= 0.25", NumericRange(minimum=0.25)),
    (">= 0.50", NumericRange(minimum=0.50)),
    (">= 0.75", NumericRange(minimum=0.75)),
    (">= 0.90", NumericRange(minimum=0.90)),
    ("<= 0.25", NumericRange(maximum=0.25)),
    ("<= 0.50", NumericRange(maximum=0.50)),
    ("<= 0.75", NumericRange(maximum=0.75)),
    ("<= 0.90", NumericRange(maximum=0.90)),
)

_RANGE_PRESETS: dict[str, tuple[tuple[str, NumericRange], ...]] = {
    "prob_hierarchical_artifact_or_nonvariable": _PROBABILITY_RANGE_PRESETS,
    "prob_usable_astrophysical_variable": _PROBABILITY_RANGE_PRESETS,
    "prob_dipper_dimming": _PROBABILITY_RANGE_PRESETS,
    "prob_eb_geometric_periodic": _PROBABILITY_RANGE_PRESETS,
    "prob_long_timescale_variable": _PROBABILITY_RANGE_PRESETS,
    "prob_brightening_transient": _PROBABILITY_RANGE_PRESETS,
    "prob_other_structured_variable": _PROBABILITY_RANGE_PRESETS,
    "prob_quasi_periodic_hierarchical": _PROBABILITY_RANGE_PRESETS,
    "prob_microlensing_hierarchical": _PROBABILITY_RANGE_PRESETS,
    "prob_long_period_variable_hierarchical": _PROBABILITY_RANGE_PRESETS,
    "prob_long_term_variable_hierarchical": _PROBABILITY_RANGE_PRESETS,
    "prob_recurrent_dipper_hierarchical": _PROBABILITY_RANGE_PRESETS,
    "prob_single_dipper_hierarchical": _PROBABILITY_RANGE_PRESETS,
    "prob_quasi_periodic_given_usable": _PROBABILITY_RANGE_PRESETS,
    "prob_microlensing_given_brightening": _PROBABILITY_RANGE_PRESETS,
    "prob_long_period_variable_given_long_timescale": _PROBABILITY_RANGE_PRESETS,
    "prob_long_term_variable_given_long_timescale": _PROBABILITY_RANGE_PRESETS,
    "prob_recurrent_given_dipper": _PROBABILITY_RANGE_PRESETS,
    "prob_single_given_dipper": _PROBABILITY_RANGE_PRESETS,
    "dipper_score": (
        ("unrestricted", NumericRange()),
        (">= 5", NumericRange(5)),
        (">= 10", NumericRange(10)),
        (">= 20", NumericRange(20)),
        (">= 30", NumericRange(30)),
    ),
    "jumper_score": (
        ("unrestricted", NumericRange()),
        (">= 5", NumericRange(5)),
        (">= 10", NumericRange(10)),
        (">= 20", NumericRange(20)),
        (">= 30", NumericRange(30)),
    ),
    "q": (
        ("unrestricted", NumericRange()),
        ("<= 0.15", NumericRange(maximum=0.15)),
        ("0.15 .. 0.85", NumericRange(0.15, 0.85)),
        (">= 0.85", NumericRange(0.85)),
    ),
    "m": (
        ("unrestricted", NumericRange()),
        ("brightening <= -0.25", NumericRange(maximum=-0.25)),
        ("neutral -0.25 .. 0.25", NumericRange(-0.25, 0.25)),
        ("dimming >= 0.25", NumericRange(0.25)),
    ),
    "g_magnitude": (
        ("unrestricted", NumericRange()),
        ("<= 12", NumericRange(maximum=12)),
        ("12 .. 14", NumericRange(12, 14)),
        ("14 .. 16", NumericRange(14, 16)),
        (">= 16", NumericRange(16)),
    ),
    "period_days": (
        ("unrestricted", NumericRange()),
        ("<= 1 d", NumericRange(maximum=1)),
        ("1 .. 10 d", NumericRange(1, 10)),
        (">= 10 d", NumericRange(10)),
    ),
    "confidence": (
        ("unrestricted", NumericRange()),
        (">= 2", NumericRange(2)),
        (">= 3", NumericRange(3)),
        ("= 4", NumericRange(4, 4)),
    ),
}

_ROW_ORDER: tuple[tuple[str, str, str], ...] = (
    ("queue_state", "Queue", "choice"),
    ("signal_lane", "Signal lane", "choice"),
    (
        "heading_external_photometry",
        "PHOTOMETRY DISPLAY (ASAS-SN ALWAYS ON)",
        "heading",
    ),
    ("show_external_lightcurves", "External photometry", "choice"),
    *_EXTERNAL_PHOTOMETRY_ROWS,
    (
        "heading_external_availability",
        "PHOTOMETRY AVAILABILITY FILTERS (AND WITH ML)",
        "heading",
    ),
    *_EXTERNAL_PHOTOMETRY_AVAILABILITY_ROWS,
    ("known_objects", "Known objects", "choice"),
    ("high_ruwe", "High RUWE", "choice"),
    ("high_pm", "High PM (>100 mas/yr)", "choice"),
    ("exclude_known_neighbors", "Catalog neighbors", "choice"),
    ("exclude_dipper_contaminants", "Dipper contaminants", "choice"),
    ("exclude_failed", "Failed pipeline", "choice"),
    ("morphology_primary", "Morphology", "taxonomy"),
    ("physical_primary", "Physical hypothesis", "taxonomy"),
    ("categorical_logic", "Class-filter logic", "choice"),
    ("heading_ml_gate", "ML GATE", "heading"),
    ("prob_hierarchical_artifact_or_nonvariable", "ML P(reject)", "range"),
    ("prob_usable_astrophysical_variable", "ML P(usable)", "range"),
    ("heading_ml_primary", "ML GLOBAL PRIMARY", "heading"),
    ("prob_dipper_dimming", "ML P(dipper)", "range"),
    ("prob_eb_geometric_periodic", "ML P(EB)", "range"),
    ("prob_long_timescale_variable", "ML P(long)", "range"),
    ("prob_brightening_transient", "ML P(bright)", "range"),
    ("prob_other_structured_variable", "ML P(other)", "range"),
    ("heading_ml_global_subtypes", "ML GLOBAL SUBTYPES", "heading"),
    ("prob_quasi_periodic_hierarchical", "ML P(QP)", "range"),
    ("prob_microlensing_hierarchical", "ML P(micro)", "range"),
    ("prob_long_period_variable_hierarchical", "ML P(LPV)", "range"),
    ("prob_long_term_variable_hierarchical", "ML P(LTV)", "range"),
    ("prob_recurrent_dipper_hierarchical", "ML P(recurrent)", "range"),
    ("prob_single_dipper_hierarchical", "ML P(single)", "range"),
    ("heading_ml_conditional", "ML CONDITIONAL HEADS", "heading"),
    ("prob_quasi_periodic_given_usable", "ML P(QP | usable)", "range"),
    (
        "prob_microlensing_given_brightening",
        "ML P(micro | bright)",
        "range",
    ),
    (
        "prob_long_period_variable_given_long_timescale",
        "ML P(LPV | long)",
        "range",
    ),
    (
        "prob_long_term_variable_given_long_timescale",
        "ML P(LTV | long)",
        "range",
    ),
    ("prob_recurrent_given_dipper", "ML P(recur | dip)", "range"),
    ("prob_single_given_dipper", "ML P(single | dip)", "range"),
    ("heading_ml_predictions", "ML PREDICTED CLASSES", "heading"),
    ("predicted_hierarchy_gate", "ML gate class", "choice"),
    ("predicted_primary_morphology", "ML primary | usable", "choice"),
    ("predicted_hierarchical_class", "ML hierarchy class", "choice"),
    ("predicted_quasi_periodic", "ML QP class", "choice"),
    ("predicted_microlensing_like", "ML microlens class", "choice"),
    ("predicted_long_timescale_subtype", "ML long subtype", "choice"),
    ("predicted_dipper_recurrence", "ML recurrence class", "choice"),
    ("heading_triage_metrics", "TRIAGE METRICS", "heading"),
    ("dipper_score", "Dipper score", "range"),
    ("jumper_score", "Jumper score", "range"),
    ("q", "Q", "range"),
    ("m", "M", "range"),
    ("g_magnitude", "G magnitude", "range"),
    ("period_days", "Period", "range"),
    ("confidence", "Confidence", "range"),
    ("sort_by", "Sort", "choice"),
    ("sort_desc", "Direction", "choice"),
)

_VALUE_LABELS = {
    "followup": "follow-up",
    "any": "any",
    "exclude": "exclude",
    "only": "only",
    "candidate_id": "candidate ID",
    "prob_hierarchical_artifact_or_nonvariable": "ML reject gate",
    "prob_usable_astrophysical_variable": "ML usable gate",
    "prob_dipper_dimming": "ML dipper / dimming",
    "prob_eb_geometric_periodic": "ML EB / geometric",
    "prob_long_timescale_variable": "ML long-timescale",
    "prob_brightening_transient": "ML brightening",
    "prob_other_structured_variable": "ML other structured",
    "prob_quasi_periodic_hierarchical": "ML quasi-periodic",
    "prob_microlensing_hierarchical": "ML microlensing",
    "prob_long_period_variable_hierarchical": "ML LPV",
    "prob_long_term_variable_hierarchical": "ML LTV",
    "prob_recurrent_dipper_hierarchical": "ML recurrent dipper",
    "prob_single_dipper_hierarchical": "ML single dipper",
    "prob_quasi_periodic_given_usable": "ML QP | usable",
    "prob_microlensing_given_brightening": "ML microlens | brightening",
    "prob_long_period_variable_given_long_timescale": "ML LPV | long",
    "prob_long_term_variable_given_long_timescale": "ML LTV | long",
    "prob_recurrent_given_dipper": "ML recurrent | dipper",
    "prob_single_given_dipper": "ML single | dipper",
    "artifact_or_nonvariable": "artifact / nonvariable",
    "usable_astrophysical_variable": "usable variable",
    "dipper_dimming": "dipper / dimming",
    "eb_geometric_periodic": "EB / geometric",
    "long_timescale_variable": "long-timescale",
    "brightening_transient": "brightening / transient",
    "other_structured_variable": "other structured",
    "quasi_periodic": "quasi-periodic",
    "not_quasi_periodic": "not quasi-periodic",
    "microlensing_like": "microlensing-like",
    "not_microlensing_like": "not microlensing-like",
    "long_period_variable": "LPV",
    "long_term_variable": "LTV",
    "recurrent": "recurrent",
    "non_recurrent": "single / non-recurrent",
    "not_applicable": "not applicable",
    "dipper_score": "dipper score",
    "jumper_score": "jumper score",
    "g_magnitude": "G magnitude",
    "period_days": "period",
    "last_review": "last review time",
    "all": "ALL",
}


class FilterEditor:
    """Mutable cursor around an immutable :class:`QueueFilterSpec`."""

    def __init__(
        self,
        spec: QueueFilterSpec,
        *,
        catalogs: Iterable[str] | None = None,
        external_photometry_counts: dict[str, int] | None = None,
    ) -> None:
        self.spec = spec
        self.cursor = 0
        self.external_photometry_counts = {
            str(source): max(0, int(count))
            for source, count in (external_photometry_counts or {}).items()
        }
        valid_catalogs = {catalog for catalog, _label, _column in CATALOG_TYPE_SOURCES}
        requested_catalogs = (
            valid_catalogs
            if catalogs is None
            else {str(catalog) for catalog in catalogs}
        )
        selected_catalogs = valid_catalogs & requested_catalogs
        catalog_rows = tuple(
            (f"catalog_{catalog}", f"{label} types", "catalog")
            for catalog, label, _column in CATALOG_TYPE_SOURCES
            if catalog in selected_catalogs
        )
        catalog_insert_index = next(
            index + 1
            for index, (key, _label, _kind) in enumerate(_ROW_ORDER)
            if key == "known_objects"
        )
        self._row_order = (
            *_ROW_ORDER[:catalog_insert_index],
            *catalog_rows,
            *_ROW_ORDER[catalog_insert_index:],
        )

    @property
    def row_count(self) -> int:
        return len(self._row_order)

    @property
    def active_key(self) -> str:
        return self._row_order[self.cursor][0]

    @property
    def active_kind(self) -> str:
        return self._row_order[self.cursor][2]

    @property
    def active_catalog(self) -> str | None:
        if self.active_kind != "catalog":
            return None
        return self.active_key.removeprefix("catalog_")

    def move(self, delta: int) -> None:
        selectable = [
            index
            for index, (_key, _label, kind) in enumerate(self._row_order)
            if kind != "heading"
        ]
        position = selectable.index(self.cursor)
        position = min(max(position + int(delta), 0), len(selectable) - 1)
        self.cursor = selectable[position]

    def reset(self, *, only_unreviewed: bool = True) -> None:
        self.spec = QueueFilterSpec.default(only_unreviewed=only_unreviewed)
        self.cursor = 0

    def cycle(self, delta: int = 1) -> None:
        key = self.active_key
        if self.active_kind in {"taxonomy", "catalog", "heading"}:
            return
        if self.active_kind == "external_source":
            self.toggle_external_source(key.removeprefix("external_source_"))
            return
        if self.active_kind == "external_availability":
            source = key.removeprefix("external_availability_")
            current = self.external_availability_mode(source)
            index = _EXTERNAL_AVAILABILITY_MODES.index(current)
            mode = _EXTERNAL_AVAILABILITY_MODES[
                (index + int(delta)) % len(_EXTERNAL_AVAILABILITY_MODES)
            ]
            self.set_external_availability_mode(source, mode)
            return
        choices = (
            tuple(value for _label, value in _RANGE_PRESETS[key])
            if self.active_kind == "range"
            else _CHOICES[key]
        )
        current = getattr(self.spec, key)
        try:
            index = choices.index(current)
        except ValueError:
            index = 0
        value = choices[(index + int(delta)) % len(choices)]
        self.spec = replace(self.spec, **{key: value})

    def external_source_enabled(self, source: str) -> bool:
        """Return whether one supported external source is selected."""

        return str(source) in self.spec.external_lightcurve_sources

    def toggle_external_source(self, source: str) -> bool:
        """Toggle one external magnitude source while preserving display order."""

        source_key = str(source or "").strip().lower()
        if source_key not in TUI_EXTERNAL_PHOTOMETRY_SOURCE_CHOICES:
            raise ValueError(f"unknown external photometry source {source!r}")
        selected = set(self.spec.external_lightcurve_sources)
        if source_key in selected:
            selected.remove(source_key)
            enabled = False
        else:
            selected.add(source_key)
            enabled = True
        ordered = tuple(
            value
            for value in TUI_EXTERNAL_PHOTOMETRY_SOURCE_CHOICES
            if value in selected
        )
        self.spec = replace(
            self.spec,
            external_lightcurve_sources=ordered,
        )
        return enabled

    def external_availability_mode(self, source: str) -> str:
        """Return any/required/absent for one external photometry source."""

        source_key = str(source or "").strip().lower()
        if source_key not in TUI_EXTERNAL_PHOTOMETRY_AVAILABILITY_SOURCE_CHOICES:
            raise ValueError(f"unknown external photometry source {source!r}")
        if source_key in self.spec.required_external_photometry_sources:
            return "required"
        if source_key in self.spec.excluded_external_photometry_sources:
            return "absent"
        return "any"

    def set_external_availability_mode(self, source: str, mode: str) -> None:
        """Set one source gate without changing any display-overlay choice."""

        source_key = str(source or "").strip().lower()
        normalized_mode = str(mode or "").strip().lower()
        if source_key not in TUI_EXTERNAL_PHOTOMETRY_AVAILABILITY_SOURCE_CHOICES:
            raise ValueError(f"unknown external photometry source {source!r}")
        if normalized_mode not in _EXTERNAL_AVAILABILITY_MODES:
            raise ValueError(
                "external photometry availability must be any, required, or absent"
            )
        required = set(self.spec.required_external_photometry_sources)
        excluded = set(self.spec.excluded_external_photometry_sources)
        required.discard(source_key)
        excluded.discard(source_key)
        if normalized_mode == "required":
            required.add(source_key)
        elif normalized_mode == "absent":
            excluded.add(source_key)
        ordered_required = tuple(
            value
            for value in TUI_EXTERNAL_PHOTOMETRY_AVAILABILITY_SOURCE_CHOICES
            if value in required
        )
        ordered_excluded = tuple(
            value
            for value in TUI_EXTERNAL_PHOTOMETRY_AVAILABILITY_SOURCE_CHOICES
            if value in excluded
        )
        self.spec = replace(
            self.spec,
            required_external_photometry_sources=ordered_required,
            excluded_external_photometry_sources=ordered_excluded,
        )

    def toggle_taxonomy(self, value: str) -> bool:
        key = self.active_key
        if key not in {"morphology_primary", "physical_primary"}:
            raise ValueError("active filter row is not a taxonomy selector")
        selected = list(getattr(self.spec, key))
        if value in selected:
            selected.remove(value)
            enabled = False
        else:
            selected.append(value)
            enabled = True
        self.spec = replace(self.spec, **{key: tuple(selected)})
        return enabled

    def clear_taxonomy(self) -> None:
        if self.active_key in {"morphology_primary", "physical_primary"}:
            self.spec = replace(self.spec, **{self.active_key: ()})

    def catalog_type_kept(self, catalog: str, value: str) -> bool:
        field_name = CATALOG_TYPE_EXCLUSION_FIELDS.get(str(catalog))
        if field_name is None:
            raise ValueError(f"unknown catalog {catalog!r}")
        return str(value) not in getattr(self.spec, field_name)

    def set_catalog_type_kept(
        self,
        catalog: str,
        value: str,
        kept: bool,
    ) -> bool:
        field_name = CATALOG_TYPE_EXCLUSION_FIELDS.get(str(catalog))
        if field_name is None:
            raise ValueError(f"unknown catalog {catalog!r}")
        text = str(value or "").strip()
        if not text:
            raise ValueError("catalog type must not be blank")
        excluded = list(getattr(self.spec, field_name))
        if kept:
            excluded = [item for item in excluded if item != text]
        elif text not in excluded:
            excluded.append(text)
        self.spec = replace(self.spec, **{field_name: tuple(excluded)})
        return kept

    def toggle_catalog_type(self, catalog: str, value: str) -> bool:
        kept = not self.catalog_type_kept(catalog, value)
        return self.set_catalog_type_kept(catalog, value, kept)

    def keep_all_catalog_types(self, catalog: str | None = None) -> None:
        if catalog is None:
            updates = {
                field_name: ()
                for field_name in CATALOG_TYPE_EXCLUSION_FIELDS.values()
            }
        else:
            field_name = CATALOG_TYPE_EXCLUSION_FIELDS.get(str(catalog))
            if field_name is None:
                raise ValueError(f"unknown catalog {catalog!r}")
            updates = {field_name: ()}
        self.spec = replace(self.spec, **updates)

    def rows(self) -> tuple[FilterRow, ...]:
        return tuple(
            FilterRow(
                key,
                label,
                "" if kind == "heading" else self._display_value(key, kind),
                kind,
            )
            for key, label, kind in self._row_order
        )

    def _display_value(self, key: str, kind: str) -> str:
        if kind == "external_source":
            source = key.removeprefix("external_source_")
            return "enabled" if self.external_source_enabled(source) else "disabled"
        if kind == "external_availability":
            source = key.removeprefix("external_availability_")
            mode = self.external_availability_mode(source)
            count = self.external_photometry_counts.get(source)
            return mode if count is None else f"{mode} · n={count:,}"
        if kind == "catalog":
            catalog = key.removeprefix("catalog_")
            field_name = CATALOG_TYPE_EXCLUSION_FIELDS[catalog]
            count = len(getattr(self.spec, field_name))
            return "all kept" if count == 0 else f"{count} excluded"
        value = getattr(self.spec, key)
        if kind == "taxonomy":
            values = tuple(value)
            if not values:
                return "any"
            labels = [label_for(item) for item in values]
            return labels[0] if len(labels) == 1 else f"{labels[0]} +{len(labels) - 1}"
        if kind == "range":
            for label, preset in _RANGE_PRESETS[key]:
                if preset == value:
                    return label
            minimum = value.minimum
            maximum = value.maximum
            if minimum is not None and maximum is not None:
                return f"{minimum:g} .. {maximum:g}"
            if minimum is not None:
                return f">= {minimum:g}"
            if maximum is not None:
                return f"<= {maximum:g}"
            return "unrestricted"
        if isinstance(value, bool):
            if key == "sort_desc":
                return "descending" if value else "ascending"
            if key == "show_external_lightcurves":
                return "enabled" if value else "disabled"
            return "exclude" if value else "any"
        return _VALUE_LABELS.get(str(value), str(value))


__all__ = ["FilterEditor", "FilterRow"]
