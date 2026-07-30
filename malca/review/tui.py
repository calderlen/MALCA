"""Minimal keyboard-driven terminal reviewer for MALCA candidates."""

from __future__ import annotations

import argparse
import curses
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import tempfile
import textwrap
import time
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING
import webbrowser

from malca.config import POST_FILTER_MAX_PM
from malca.review.taxonomy import label_for
from malca.review.tui_controller import (
    MORPHOLOGY_PRIMARY_BY_KEY,
    MORPHOLOGY_PRIMARY_ITEMS,
    MORPHOLOGY_SECONDARY_BY_KEY,
    MORPHOLOGY_SECONDARY_ITEMS,
    PHYSICAL_PRIMARY_BY_KEY,
    PHYSICAL_PRIMARY_ITEMS,
    PHYSICAL_SECONDARY_BY_KEY,
    PHYSICAL_SECONDARY_ITEMS,
    ML_CLASS_SCORE_FIELDS,
    MenuItem,
    ReviewDraft,
    compact_detail_lines,
    detail_sections,
    build_review_identity_line,
    _format_period,
)
from malca.review.tui_photometry import (
    normalize_tui_external_photometry_sources,
    tui_external_photometry_source_label,
)

if TYPE_CHECKING:
    from malca.review.tui_render import ImageCoordinator, RenderRequest
    from malca.review.tui_service import (
        CandidateRecord,
        CatalogTypeStat,
        ReviewRepository,
    )


MIN_TERMINAL_HEIGHT = 32
MIN_TERMINAL_WIDTH = 48
PORTRAIT_MAX_WIDTH = 99
PORTRAIT_FOOTER_HEIGHT = 6
LANDSCAPE_FOOTER_HEIGHT = 5
IMAGE_POLL_MS = 100
SAVE_DEBOUNCE_SECONDS = 0.75


def _ellipsize_text(value: object, width: int) -> str:
    limit = max(0, int(width))
    text = str(value)
    if len(text) <= limit:
        return text
    if limit <= 0:
        return ""
    if limit == 1:
        return "…"
    return text[: limit - 1].rstrip() + "…"


def _compact_token_line(parts: Sequence[object], width: int) -> str:
    """Fit a filter summary without chopping the final sort token."""

    limit = max(1, int(width))
    tokens = [str(part).strip() for part in parts if str(part).strip()]
    if not tokens:
        return "—"
    joined = " · ".join(tokens)
    if len(joined) <= limit:
        return joined
    if len(tokens) == 1:
        return _ellipsize_text(tokens[0], limit)

    first, last = tokens[0], tokens[-1]
    middle = list(tokens[1:-1])
    omitted = 0
    while True:
        candidate = [first, *middle]
        if omitted:
            candidate.append(f"+{omitted}")
        candidate.append(last)
        text = " · ".join(candidate)
        if len(text) <= limit:
            return text
        if middle:
            middle.pop()
            omitted += 1
            continue
        break

    suffix = f" · +{max(omitted, 1)} · {last}"
    return _ellipsize_text(first, max(1, limit - len(suffix))) + suffix


def _short_phase_source(source: object) -> str:
    """Compact fold-method label for the portrait context block."""

    text = str(source or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    if "harmonic" in lowered:
        return "harmonic"
    if "pipeline" in lowered:
        return "pipeline"
    if "manual" in lowered:
        return "manual"
    if "pdm" in lowered:
        return "PDM"
    if "unavailable" in lowered:
        return "unavailable"
    if text.startswith("Auto "):
        return text[5:]
    return text


def _detail_value_map(sections: Sequence[object]) -> dict[str, str]:
    """Extract aligned label/value pairs from controller detail sections."""

    values: dict[str, str] = {}
    for section in sections:
        for raw_line in getattr(section, "lines", ()):
            line = str(raw_line)
            content = line[2:] if line.startswith("  ") else line
            label = content[:9].strip()
            value = content[10:].strip() if len(content) > 9 else ""
            if label:
                values[label] = value or "—"
    return values


def _is_missing_detail_value(value: object) -> bool:
    text = str(value or "").strip()
    return text in {"", "—", "-", "none", "nan", "null", "unknown"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="malca review-tui",
        description="Review MALCA candidates with one persistent image window.",
    )
    parser.add_argument(
        "--review-db",
        required=True,
        type=Path,
        help="Review SQLite database path.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run root containing bundle_assets/lightcurves (normally inferred from the DB).",
    )
    parser.add_argument(
        "--candidate",
        default=None,
        help="Candidate ID or ASAS-SN ID to open initially.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include reviewed candidates (default: snapshot only unreviewed candidates).",
    )
    parser.add_argument(
        "--reviewer",
        default="calder",
        help="Reviewer name stored with each save (default: calder).",
    )
    parser.add_argument(
        "--viewer",
        choices=("window", "quicklook", "none"),
        default="window",
        help="Image viewer (default: persistent window; preserves its size and position).",
    )
    parser.add_argument(
        "--image-cache-size",
        type=int,
        default=4,
        help="Number of rendered candidate PNGs to retain (default: 4).",
    )
    parser.add_argument(
        "--external-time-window",
        choices=("full", "asassn"),
        default="full",
        help=(
            "Initial raw-panel time window: full external baseline or the "
            "ASAS-SN light-curve span (default: full)."
        ),
    )
    parser.add_argument(
        "--external-time-padding-days",
        type=float,
        default=0.0,
        help=(
            "Days of padding on each side of the ASAS-SN time window "
            "(default: 0)."
        ),
    )
    return parser


@dataclass
class _ModeState:
    name: str = "primary"
    previous: str = "primary"
    page: int = 0


class ReviewTuiApp:
    """Thin curses shell over the pure draft, repository, and renderer layers."""

    def __init__(
        self,
        screen: Any,
        repository: "ReviewRepository",
        images: "ImageCoordinator",
        *,
        db_path: Path,
        run_dir: Path | None,
        external_time_window: str = "full",
        external_time_padding_days: float = 0.0,
    ) -> None:
        self.screen = screen
        self.repository = repository
        self.images = images
        self.db_path = Path(db_path)
        self.run_dir = Path(run_dir).expanduser() if run_dir is not None else None
        self.index = int(repository.start_index)
        self.record: "CandidateRecord | None" = None
        self.review: dict[str, Any] = {}
        self.draft = ReviewDraft.from_review({})
        self.mode = _ModeState()
        # Imported only for a real app instance so ``malca review-tui --help``
        # retains the lightweight startup path.
        from malca.review.tui_filters import FilterEditor
        from malca.review.tui_service import QueueFilterSpec

        initial_filter = getattr(repository, "filter_spec", None)
        if initial_filter is None:
            initial_filter = QueueFilterSpec.default()
        self.filter_editor = FilterEditor(initial_filter)
        self.filter_match_count = len(getattr(repository, "candidate_ids", ()))
        self.catalog_type_stats: tuple[CatalogTypeStat, ...] = ()
        self.catalog_type_cursor = 0
        self.input_buffer = ""
        self._pending_action: tuple[str, object] | None = None
        self.phase_multiplier = 1.0
        self.manual_phase_period_days: float | None = None
        self.manual_phase_source: str | None = None
        self.force_period_search = False
        self.force_period_search_token: str | None = None
        self.phase_period_days: float | None = None
        self.phase_source = ""
        self.show_event_markers = False
        self.camera_view = "cleaned"
        self.color_by_camera = False
        self.show_external_lightcurves = bool(
            initial_filter.show_external_lightcurves
        )
        self.external_lightcurve_sources = (
            normalize_tui_external_photometry_sources(
                initial_filter.external_lightcurve_sources
            )
        )
        self.time_window_mode = (
            str(external_time_window).strip().lower()
            if str(external_time_window).strip().lower() in {"full", "asassn"}
            else "full"
        )
        try:
            padding_days = float(external_time_padding_days)
        except (TypeError, ValueError):
            padding_days = 0.0
        self.asassn_window_padding_days = (
            padding_days
            if math.isfinite(padding_days) and padding_days >= 0.0
            else 0.0
        )
        self.plot_theme = "black"
        # Default to adaptive review bounds (index 0); ``W`` cycles presets.
        from malca.review.tui_render import PHASE_SEARCH_WINDOWS

        self._phase_search_windows = tuple(PHASE_SEARCH_WINDOWS)
        self._phase_search_window_index = 0
        self.survey_label = "DECaPS DR2"
        self.notice = ""
        self.notice_error = False
        self.image_status = "waiting"
        self._current_image_generation: int | None = None
        self._image_gate_notice = False
        self._last_save_completed_at = float("-inf")
        self._last_saved_candidate_id: str | None = None
        self._running = True
        self._styles: dict[str, int] = {}
        self.vsx_url: str | None = None

    @property
    def candidate_ids(self) -> Sequence[str]:
        return self.repository.candidate_ids

    @property
    def candidate_id(self) -> str | None:
        if not self.candidate_ids:
            return None
        if self.index < 0 or self.index >= len(self.candidate_ids):
            return None
        return str(self.candidate_ids[self.index])

    def run(self) -> None:
        self._configure_screen()
        try:
            if self.candidate_ids:
                self._load_current("")
            else:
                self.notice = "Queue is empty; press F to change or reset filters."
                self.notice_error = True
            while self._running:
                self._poll_images()
                self._draw()
                try:
                    key = self.screen.get_wch()
                except curses.error:
                    continue
                self._handle_key(key)
        finally:
            self._persist_position_best_effort()

    def _configure_screen(self) -> None:
        self.screen.keypad(True)
        self.screen.timeout(IMAGE_POLL_MS)
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        # Styles use color, not weight, to distinguish semantic categories.
        # Bold is avoided everywhere it isn't a genuine visual signal because
        # bold columns quietly steal cells in dense terminal layouts.
        self._styles = {
            "title": 0,
            "section": 0,
            "label": curses.A_DIM,
            "value": 0,
            "key": 0,
            "catalog": 0,
            "selected": curses.A_REVERSE,
            "dirty": 0,
            "dim": curses.A_DIM,
            "error": 0,
            "link": curses.A_UNDERLINE,
        }
        if curses.has_colors():
            try:
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_CYAN, -1)
                curses.init_pair(2, curses.COLOR_GREEN, -1)
                curses.init_pair(3, curses.COLOR_YELLOW, -1)
                curses.init_pair(4, curses.COLOR_RED, -1)
                curses.init_pair(5, curses.COLOR_MAGENTA, -1)
                self._styles.update(
                    title=curses.color_pair(1),
                    section=curses.color_pair(1),
                    key=curses.color_pair(1),
                    catalog=curses.color_pair(2),
                    selected=curses.color_pair(2) | curses.A_REVERSE,
                    dirty=curses.color_pair(3),
                    error=curses.color_pair(4),
                    link=curses.A_UNDERLINE | curses.color_pair(1),
                )
            except curses.error:
                pass

    def _load_current(self, notice: str = "") -> None:
        candidate_id = self.candidate_id
        if candidate_id is None:
            return
        self.record, self.review = self.repository.load(candidate_id)
        self.draft = ReviewDraft.from_review(self.review)
        self.vsx_url = self._external_link("VSX")
        self.mode = _ModeState()
        self.phase_multiplier = 1.0
        self.manual_phase_period_days = None
        self.manual_phase_source = None
        self.force_period_search = False
        self.force_period_search_token = None
        self.phase_period_days = None
        self.phase_source = ""
        self.survey_label = "DECaPS DR2"
        self.notice = notice
        self.notice_error = False
        self._request_current_image()

    def _external_link(self, label: str) -> str | None:
        payload = getattr(self.record, "payload", None)
        if not isinstance(payload, dict):
            return None
        try:
            from malca.review.metadata import build_external_lookup_links

            return next(
                (url for item_label, url in build_external_lookup_links(payload)
                 if item_label == label),
                None,
            )
        except Exception:
            return None

    def _render_request(
        self,
        record: "CandidateRecord",
        *,
        use_display_controls: bool = True,
    ) -> "RenderRequest":
        from malca.core.phase import resolve_phase_period
        from malca.review.period_search import (
            adaptive_review_period_bounds,
            resolve_stored_review_period,
        )
        from malca.review.tui_render import RenderRequest

        raw_payload = getattr(record, "payload", None)
        payload = dict(raw_payload) if isinstance(raw_payload, dict) else None
        phase_period, phase_source = resolve_stored_review_period(payload)
        search_min, search_max = self._phase_search_bounds(payload)
        request = RenderRequest(
            candidate_id=record.candidate_id,
            asas_sn_id=record.asas_sn_id,
            lc_path=record.lc_path,
            source_path=record.source_path,
            db_path=self.db_path,
            run_dir=self.run_dir,
            payload=payload,
            stored_phase_period_days=phase_period,
            stored_phase_source=phase_source or None,
            manual_phase_period_days=(
                self.manual_phase_period_days if use_display_controls else None
            ),
            manual_phase_source=(
                self.manual_phase_source if use_display_controls else None
            ),
            phase_multiplier=self.phase_multiplier if use_display_controls else 1.0,
            force_period_search=(
                self.force_period_search if use_display_controls else False
            ),
            force_period_search_token=(
                self.force_period_search_token
                if use_display_controls and self.force_period_search
                else None
            ),
            # These are session-level view preferences, so prefetch must use
            # them too.  Only phase overrides are candidate-specific.
            camera_view=self.camera_view,
            show_event_markers=self.show_event_markers,
            color_by_camera=self.color_by_camera,
            show_external_lightcurves=self.show_external_lightcurves,
            external_lightcurve_sources=self.external_lightcurve_sources,
            time_window_mode=self.time_window_mode,
            asassn_window_padding_days=self.asassn_window_padding_days,
            plot_theme=self.plot_theme,
            survey_key="decaps-dr2",
            phase_search_min_days=search_min,
            phase_search_max_days=search_max,
        )
        if use_display_controls:
            base_period = self.manual_phase_period_days or phase_period
            if base_period is not None and not self.force_period_search:
                self.phase_period_days = float(base_period) * self.phase_multiplier
                if self.manual_phase_period_days:
                    source = self.manual_phase_source or "Manual"
                else:
                    source = phase_source or "stored"
                if self.phase_multiplier != 1.0:
                    source = f"{source} x{self.phase_multiplier:g}"
                self.phase_source = source
            elif self.force_period_search:
                self.phase_period_days = None
                self.phase_source = "Pipeline search (running)"
        return request

    def _request_current_image(self) -> None:
        if self.record is None:
            return
        self.image_status = "loading"
        self._image_gate_notice = False
        status = self.images.request_current(self._render_request(self.record))
        self._current_image_generation = status.generation
        self.image_status = status.state
        next_index = self.index + 1
        if next_index < len(self.candidate_ids):
            try:
                next_record, _ = self.repository.load(str(self.candidate_ids[next_index]))
                self.images.prefetch(
                    self._render_request(next_record, use_display_controls=False)
                )
            except Exception:
                pass

    def _poll_images(self) -> None:
        result = self.images.poll()
        if result is None:
            return
        generation = getattr(result, "generation", None)
        if generation is not None and generation != self._current_image_generation:
            return
        error = getattr(result, "error", None)
        if error:
            self.image_status = f"error: {error}"
            return
        self.image_status = str(getattr(result, "state", "ready") or "ready")
        rendered_period = getattr(result, "phase_period_days", None)
        try:
            rendered_period = float(rendered_period)
        except (TypeError, ValueError):
            rendered_period = None
        valid_period = bool(
            rendered_period is not None
            and math.isfinite(rendered_period)
            and rendered_period > 0
        )
        rendered_source = str(getattr(result, "phase_source", "") or "").strip()
        if self.image_status == "ready":
            if valid_period:
                assert rendered_period is not None
                self.phase_period_days = rendered_period
                if self.manual_phase_period_days is None:
                    # Retain the expensive automatic/harmonic result as this
                    # candidate's display-period base.  Half/double, marker
                    # toggles, and cache misses then avoid another search.
                    self.manual_phase_period_days = (
                        rendered_period / self.phase_multiplier
                    )
                    self.manual_phase_source = rendered_source or "Resolved period"
                if rendered_source:
                    self.phase_source = rendered_source
            else:
                self.phase_period_days = None
                self.phase_source = rendered_source or "Auto period unavailable"
            self.force_period_search = False
            self.force_period_search_token = None
        survey_label = str(getattr(result, "survey_label", "") or "").strip()
        if survey_label:
            self.survey_label = survey_label
        if self.image_status == "ready" and self._image_gate_notice:
            # The image gate was blocking classification but is now resolved.
            # Clear the block silently rather than announcing the obvious.
            self._image_gate_notice = False
            if self.notice_error:
                self.notice = ""
                self.notice_error = False

    def _handle_key(self, key: object) -> None:
        if key == curses.KEY_MOUSE:
            return
        if self.mode.name == "help":
            self.mode.name = self.mode.previous
            return
        if self.mode.name == "quit":
            self._handle_quit_confirmation(key)
            return
        if self.mode.name == "filters":
            self._handle_filter_key(key)
            return
        if self.mode.name == "filter_catalog_types":
            self._handle_catalog_type_filter_key(key)
            return
        if self.mode.name in {"filter_morphology", "filter_physical"}:
            self._handle_filter_taxonomy_key(key)
            return
        if self.mode.name in {"filter_confirm", "search_confirm"}:
            self._handle_pending_confirmation(key)
            return
        if self.mode.name in {"search_input", "period_input", "notes_input"}:
            self._handle_text_input(key)
            return

        if key == "Q":
            self._request_quit()
            return
        if key == "?":
            self.mode.previous = self.mode.name
            self.mode.name = "help"
            return
        if key == "F":
            self._open_filters()
            return
        if key == "H":
            self.mode.name = "physical"
            self.mode.page = 0
            self._set_notice("Physical hypothesis menu")
            return
        if key == "M":
            self.mode.previous = self.mode.name
            self.mode.name = "notes_input"
            self.input_buffer = self.draft.notes
            self._set_notice("Edit review notes; Enter applies to the draft")
            return
        if key == "V":
            self._open_vsx()
            return
        if key == "/":
            self.mode.previous = self.mode.name
            self.mode.name = "search_input"
            self.input_buffer = ""
            self._set_notice("Enter a candidate or ASAS-SN ID")
            return
        if key == "P":
            self.mode.previous = self.mode.name
            self.mode.name = "period_input"
            self.input_buffer = (
                f"{self.phase_period_days:.9g}" if self.phase_period_days else ""
            )
            self._set_notice("Enter a positive display period in days")
            return
        if key == "R":
            self.manual_phase_period_days = None
            self.manual_phase_source = None
            self.phase_multiplier = 1.0
            self.force_period_search = True
            self.force_period_search_token = str(time.time_ns())
            self._set_notice("Recomputing pipeline period search")
            self._request_current_image()
            return
        if key in {"+", "="}:
            self.phase_multiplier *= 2.0
            self._set_notice("Displaying double the base period")
            self._request_current_image()
            return
        if key == "-":
            self.phase_multiplier /= 2.0
            self._set_notice("Displaying half the base period")
            self._request_current_image()
            return
        if key == "E":
            self.show_event_markers = not self.show_event_markers
            self._set_notice(
                f"Event markers: {'on' if self.show_event_markers else 'off'}"
            )
            self._request_current_image()
            return
        if key == "A":
            self.camera_view = "all" if self.camera_view == "cleaned" else "cleaned"
            if self.manual_phase_source != "Manual":
                # Automatic periods depend on the active camera cleaning.
                self.manual_phase_period_days = None
                self.manual_phase_source = None
            self._set_notice(f"Camera view: {self.camera_view}")
            self._request_current_image()
            return
        if key in {"C", "G"}:
            self.color_by_camera = not self.color_by_camera
            self._set_notice(
                f"Camera colors: {'on' if self.color_by_camera else 'off'}"
            )
            self._request_current_image()
            return
        if key == "O":
            self.show_external_lightcurves = not self.show_external_lightcurves
            source_count = len(self.external_lightcurve_sources)
            self._set_notice(
                "External photometry: "
                + (
                    f"on ({source_count} source{'s' if source_count != 1 else ''})"
                    if self.show_external_lightcurves
                    else "off"
                )
            )
            self._request_current_image()
            return
        if key == "L":
            self.time_window_mode = (
                "asassn" if self.time_window_mode == "full" else "full"
            )
            self._set_notice(f"Light-curve window: {self._time_window_label()}")
            self._request_current_image()
            return
        if key == "T":
            from malca.review.tui_render import TUI_PLOT_THEME_CYCLE

            index = (
                TUI_PLOT_THEME_CYCLE.index(self.plot_theme)
                if self.plot_theme in TUI_PLOT_THEME_CYCLE
                else -1
            )
            self.plot_theme = TUI_PLOT_THEME_CYCLE[(index + 1) % len(TUI_PLOT_THEME_CYCLE)]
            self._set_notice(f"Plot theme: {self.plot_theme}")
            self._request_current_image()
            return
        if key == "W":
            # Cycle the PDM search window and force an automatic re-fit so the
            # phase panel is re-computed against the new range on the next
            # render.
            self._phase_search_window_index = (
                self._phase_search_window_index + 1
            ) % len(self._phase_search_windows)
            self.manual_phase_period_days = None
            self.manual_phase_source = None
            self.phase_multiplier = 1.0
            self.force_period_search = True
            self.force_period_search_token = str(time.time_ns())
            self._set_notice(
                f"PDM window: {self._phase_search_window_label()}"
            )
            self._request_current_image()
            return
        if key == "X":
            self._export_current_render()
            return
        if self._is_page_down(key):
            self.mode.page += 1
            return
        if self._is_page_up(key):
            self.mode.page = max(0, self.mode.page - 1)
            return
        if key == "N" or self._is_tab(key):
            self._navigate(1, "Skipped without saving")
            return
        if self._is_candidate_edit_key(key) and not self._image_is_ready():
            self._block_for_image()
            return

        if isinstance(key, str) and key in {"1", "2", "3", "4"}:
            self.draft.set_confidence(int(key))
            self._set_notice(f"Confidence: {key}")
            return
        if key == ",":
            self.draft.toggle_followup()
            self._set_notice(
                "Follow-up: ON" if self.draft.needs_followup else "Follow-up: OFF"
            )
            return
        if key == "S" or self._is_enter(key):
            self._save(advance=True)
            return
        if key == ".":
            self._save(advance=False)
            return
        if self.mode.name == "subtypes":
            self._handle_subtype_key(key)
        elif self.mode.name == "physical_subtypes":
            self._handle_physical_subtype_key(key)
        elif self.mode.name == "physical":
            self._handle_physical_key(key)
        else:
            self._handle_primary_key(key)

    def _open_filters(self) -> None:
        from malca.review.tui_filters import FilterEditor

        active = getattr(self.repository, "filter_spec", self.filter_editor.spec)
        active = replace(
            active,
            show_external_lightcurves=self.show_external_lightcurves,
            external_lightcurve_sources=self.external_lightcurve_sources,
        )
        self.mode.previous = self.mode.name
        self.mode.name = "filters"
        self.mode.page = 0
        inventory_error: str | None = None
        try:
            loader = getattr(self.repository, "catalog_type_stats", None)
            self.catalog_type_stats = tuple(loader()) if callable(loader) else ()
        except Exception as exc:
            self.catalog_type_stats = ()
            inventory_error = f"Catalog-type inventory failed: {exc}"
        available_catalogs = tuple(
            dict.fromkeys(stat.catalog for stat in self.catalog_type_stats)
        )
        external_counts_loader = getattr(
            self.repository,
            "external_photometry_counts",
            None,
        )
        external_counts = (
            dict(external_counts_loader())
            if callable(external_counts_loader)
            else {}
        )
        self.filter_editor = FilterEditor(
            active,
            catalogs=available_catalogs,
            external_photometry_counts=external_counts,
        )
        self.catalog_type_cursor = 0
        self._refresh_filter_count()
        if inventory_error is not None:
            self._set_notice(inventory_error, error=True)
        elif self.filter_match_count >= 0:
            self._set_notice("Arrows adjust · Enter applies snapshot")

    def _refresh_filter_count(self) -> None:
        try:
            self.filter_match_count = int(
                self.repository.preview_filter_count(self.filter_editor.spec)
            )
        except Exception as exc:
            self.filter_match_count = -1
            self._set_notice(f"Filter preview failed: {exc}", error=True)

    def _handle_filter_key(self, key: object) -> None:
        if self._is_escape(key):
            self.mode.name = self.mode.previous
            self._set_notice("Filter changes cancelled")
            return
        if key == "R":
            self.filter_editor.reset()
            self._refresh_filter_count()
            self._set_notice("Filters reset (press Enter to apply)")
            return
        if key in {curses.KEY_UP, "k"}:
            self.filter_editor.move(-1)
            return
        if key in {curses.KEY_DOWN, "j"}:
            self.filter_editor.move(1)
            return
        if key in {curses.KEY_LEFT, "h", "["}:
            self.filter_editor.cycle(-1)
            self._refresh_filter_count()
            return
        if key in {curses.KEY_RIGHT, "l", "]"}:
            self.filter_editor.cycle(1)
            self._refresh_filter_count()
            return
        if key == " ":
            if self.filter_editor.active_kind == "taxonomy":
                self.mode.name = (
                    "filter_morphology"
                    if self.filter_editor.active_key == "morphology_primary"
                    else "filter_physical"
                )
                self.mode.page = 0
            elif self.filter_editor.active_kind == "catalog":
                self.mode.name = "filter_catalog_types"
                self.mode.page = 0
                self.catalog_type_cursor = 0
            else:
                self.filter_editor.cycle(1)
                self._refresh_filter_count()
            return
        if self._is_enter(key):
            self._pending_action = ("filters", self.filter_editor.spec)
            if self.draft.dirty:
                self.mode.name = "filter_confirm"
            else:
                self._apply_pending_action()

    def _handle_catalog_type_filter_key(self, key: object) -> None:
        if self._is_escape(key) or self._is_enter(key):
            self.mode.name = "filters"
            self.mode.page = 0
            self._refresh_filter_count()
            return
        stats = self._active_catalog_type_stats()
        if not stats:
            return
        if key in {curses.KEY_UP, "k"}:
            self.catalog_type_cursor = max(0, self.catalog_type_cursor - 1)
            return
        if key in {curses.KEY_DOWN, "j"}:
            self.catalog_type_cursor = min(
                len(stats) - 1,
                self.catalog_type_cursor + 1,
            )
            return
        if self._is_page_up(key):
            self.catalog_type_cursor = max(0, self.catalog_type_cursor - 8)
            return
        if self._is_page_down(key):
            self.catalog_type_cursor = min(
                len(stats) - 1,
                self.catalog_type_cursor + 8,
            )
            return

        stat = stats[self.catalog_type_cursor]
        if key == " ":
            kept = self.filter_editor.toggle_catalog_type(
                stat.catalog,
                stat.value,
            )
        elif isinstance(key, str) and key.lower() == "y":
            kept = self.filter_editor.set_catalog_type_kept(
                stat.catalog,
                stat.value,
                True,
            )
        elif isinstance(key, str) and key.lower() == "n":
            kept = self.filter_editor.set_catalog_type_kept(
                stat.catalog,
                stat.value,
                False,
            )
        elif key in {"A", "\x7f", "\b", curses.KEY_BACKSPACE}:
            self.filter_editor.keep_all_catalog_types(stat.catalog)
            self._refresh_filter_count()
            self._set_notice(f"All {stat.catalog_label} types set to YES")
            return
        else:
            return
        self._refresh_filter_count()
        self._set_notice(
            f"{stat.catalog_label} {stat.value}: "
            f"{'YES / keep' if kept else 'NO / exclude'}"
        )

    def _active_catalog_type_stats(self) -> tuple[CatalogTypeStat, ...]:
        catalog = self.filter_editor.active_catalog
        if catalog is None:
            return ()
        return tuple(
            stat for stat in self.catalog_type_stats if stat.catalog == catalog
        )

    def _handle_filter_taxonomy_key(self, key: object) -> None:
        if self._is_escape(key):
            self.mode.name = "filters"
            self.mode.page = 0
            self._refresh_filter_count()
            return
        if self._is_backspace(key):
            self.filter_editor.clear_taxonomy()
            self._refresh_filter_count()
            self._set_notice("Taxonomy filter cleared")
            return
        if self._is_page_down(key):
            self.mode.page += 1
            return
        if self._is_page_up(key):
            self.mode.page = max(0, self.mode.page - 1)
            return
        if not isinstance(key, str):
            return
        mapping = (
            MORPHOLOGY_PRIMARY_BY_KEY
            if self.mode.name == "filter_morphology"
            else PHYSICAL_PRIMARY_BY_KEY
        )
        item = mapping.get(key.lower())
        if item is None:
            return
        enabled = self.filter_editor.toggle_taxonomy(item.value)
        self._refresh_filter_count()
        self._set_notice(
            f"Filter {'includes' if enabled else 'removed'}: {item.label}"
        )

    def _handle_pending_confirmation(self, key: object) -> None:
        if self._is_escape(key):
            self.mode.name = "filters" if self.mode.name == "filter_confirm" else "search_input"
            return
        if not isinstance(key, str):
            return
        if key.lower() == "s":
            if self._save(advance=False):
                self._apply_pending_action()
        elif key.lower() == "d":
            self._apply_pending_action()

    def _apply_pending_action(self) -> None:
        action = self._pending_action
        if action is None:
            return
        kind, value = action
        if kind == "filters":
            try:
                self.index = int(
                    self.repository.apply_filters(
                        value,
                        anchor_candidate_id=self.candidate_id,
                    )
                )
            except Exception as exc:
                self.mode.name = "filters"
                self._set_notice(f"Filters not applied: {exc}", error=True)
                return
            self.show_external_lightcurves = bool(
                value.show_external_lightcurves
            )
            self.external_lightcurve_sources = (
                normalize_tui_external_photometry_sources(
                    value.external_lightcurve_sources
                )
            )
            self._pending_action = None
            self.filter_match_count = len(self.candidate_ids)
            self._load_current(f"Filters applied: {len(self.candidate_ids)} matches")
            return
        if kind == "search":
            self._pending_action = None
            self._perform_search(str(value))

    def _handle_text_input(self, key: object) -> None:
        if self._is_escape(key):
            self.mode.name = self.mode.previous
            self.input_buffer = ""
            self._set_notice("Input cancelled")
            return
        if self._is_backspace(key):
            self.input_buffer = self.input_buffer[:-1]
            return
        if self._is_enter(key):
            if self.mode.name == "search_input":
                query = self.input_buffer.strip()
                if not query:
                    self._set_notice("Enter a candidate or ASAS-SN ID", error=True)
                    return
                self._pending_action = ("search", query)
                if self.draft.dirty:
                    self.mode.name = "search_confirm"
                else:
                    self._apply_pending_action()
            elif self.mode.name == "period_input":
                self._apply_manual_period()
            else:
                self.draft.set_notes(self.input_buffer)
                self.mode.name = self.mode.previous
                self.input_buffer = ""
                self._set_notice("Review notes updated in draft")
            return
        if isinstance(key, str) and key.isprintable() and len(key) == 1:
            self.input_buffer += key

    def _perform_search(self, query: str) -> None:
        try:
            self.index, outside = self.repository.search_candidate(query)
        except Exception as exc:
            self.mode.name = "search_input"
            self._set_notice(f"Search failed: {exc}", error=True)
            return
        suffix = "; outside prior filters" if outside else ""
        self._load_current(f"Jumped to {self.candidate_id}{suffix}")

    def _apply_manual_period(self) -> None:
        try:
            period = float(self.input_buffer.strip())
        except ValueError:
            period = float("nan")
        if not math.isfinite(period) or period <= 0:
            self._set_notice("Period must be a positive finite number", error=True)
            return
        self.manual_phase_period_days = period
        self.manual_phase_source = "Manual"
        self.phase_multiplier = 1.0
        self.force_period_search = False
        self.force_period_search_token = None
        self.mode.name = self.mode.previous
        self._set_notice(f"Manual display period: {_format_period(period)}")
        self._request_current_image()

    def _handle_physical_key(self, key: object) -> None:
        if self._is_escape(key):
            self.mode.name = "primary"
            self.mode.page = 0
            self._set_notice("Primary morphology menu")
            return
        if self._is_backspace(key):
            self.draft.select_physical_primary(None)
            self._set_notice("Physical hypothesis cleared")
            return
        if not isinstance(key, str):
            return
        item = PHYSICAL_PRIMARY_BY_KEY.get(key.lower())
        if item is None:
            return
        if self.draft.physical_primary == item.value:
            self.draft.select_physical_primary(None)
            self.mode.name = "primary"
            self._set_notice("Physical hypothesis cleared")
            return
        self.draft.select_physical_primary(item.value)
        if PHYSICAL_SECONDARY_ITEMS.get(item.value, ()):
            self.mode.name = "physical_subtypes"
            self.mode.page = 0
            self._set_notice(f"Physical hypothesis: {item.label}; choose subtype")
        else:
            self.mode.name = "primary"
            self._set_notice(f"Physical hypothesis: {item.label}")

    def _handle_physical_subtype_key(self, key: object) -> None:
        if self._is_escape(key):
            self.mode.name = "physical"
            self.mode.page = 0
            self._set_notice("Physical hypothesis menu")
            return
        if self._is_backspace(key):
            self.draft.clear_physical_subtype()
            self.mode.name = "primary"
            self.mode.page = 0
            self._set_notice("Physical subtype cleared")
            return
        if not isinstance(key, str) or not self.draft.physical_primary:
            return
        item = PHYSICAL_SECONDARY_BY_KEY.get(
            self.draft.physical_primary, {}
        ).get(key.lower())
        if item is None:
            return
        selected = self.draft.toggle_physical_subtype(item.value)
        self._set_notice(
            f"Physical subtype {'selected' if selected else 'cleared'}: {item.label}"
        )

    def _open_vsx(self) -> None:
        if not self.vsx_url:
            self._set_notice("VSX link unavailable: candidate has no RA/Dec", error=True)
            return
        try:
            opened = webbrowser.open_new_tab(self.vsx_url)
        except Exception as exc:
            self._set_notice(f"Could not open VSX: {exc}", error=True)
            return
        if opened is False:
            self._set_notice("Could not open VSX in the default browser", error=True)
        else:
            self._set_notice("Opened VSX coordinate search")

    def _handle_primary_key(self, key: object) -> None:
        if self._is_backspace(key):
            self._navigate(-1, "Previous")
            return
        if key == " ":
            if self.draft.morphology_primary:
                self.mode.name = "subtypes"
                self.mode.page = 0
                self._set_notice("Subtype menu")
            else:
                self._set_notice("Select a primary morphology first", error=True)
            return
        if not isinstance(key, str):
            return
        item = MORPHOLOGY_PRIMARY_BY_KEY.get(key.lower())
        if item is None:
            return
        if self.draft.morphology_primary == item.value:
            self.mode.name = "subtypes"
            self.mode.page = 0
            self._set_notice("Subtype menu")
            return
        self.draft.select_primary(item.value)
        self.mode.name = "subtypes"
        self.mode.page = 0
        self._set_notice(f"Morphology: {item.label}")

    def _handle_subtype_key(self, key: object) -> None:
        if self._is_escape(key):
            self.mode.name = "primary"
            self.mode.page = 0
            self._set_notice("Primary morphology menu")
            return
        if self._is_backspace(key):
            self.draft.clear_subtypes()
            self._set_notice("Subtypes cleared")
            return
        if not isinstance(key, str) or not self.draft.morphology_primary:
            return
        item = MORPHOLOGY_SECONDARY_BY_KEY.get(
            self.draft.morphology_primary, {}
        ).get(key.lower())
        if item is None:
            return
        self.draft.toggle_subtype(item.value)
        selected = item.value in self.draft.morphology_secondaries
        self._set_notice(
            f"Subtype {'selected' if selected else 'removed'}: {item.label}"
        )

    def _save(self, *, advance: bool) -> bool:
        candidate_id = self.candidate_id
        if candidate_id is None:
            self._set_notice("Queue is empty", error=True)
            return False
        if (
            candidate_id == self._last_saved_candidate_id
            and time.monotonic() - self._last_save_completed_at
            < SAVE_DEBOUNCE_SECONDS
        ):
            self._set_notice("Save key ignored (debounce)")
            return False
        workflow = str(
            self.review.get("workflow_status")
            or self.review.get("status")
            or "unreviewed"
        ).strip().lower()
        if not self.draft.dirty and workflow != "unreviewed":
            self._set_notice("Already saved and unchanged; edit it or Tab to skip")
            return False
        errors = self.draft.validate()
        if errors:
            self._set_notice("; ".join(errors), error=True)
            return False
        try:
            refreshed = self.repository.save(
                candidate_id,
                self.draft,
                increment_pass=advance,
                event_type="tui_done" if advance else "tui_save",
            )
        except Exception as exc:
            self._set_notice(f"Save failed: {exc}", error=True)
            return False

        self.review = refreshed
        self.draft.mark_saved()
        self._last_save_completed_at = time.monotonic()
        self._last_saved_candidate_id = candidate_id
        if advance and self.index + 1 < len(self.candidate_ids):
            self.index += 1
            self._load_current("Saved + next")
        elif advance:
            self._set_notice("Saved; already at final candidate")
        else:
            self._set_notice("Saved")
        return True

    def _navigate(self, delta: int, notice: str) -> None:
        if not self.candidate_ids:
            return
        next_index = min(max(self.index + int(delta), 0), len(self.candidate_ids) - 1)
        if next_index == self.index:
            self._set_notice(
                "Already at first candidate" if delta < 0 else "Already at final candidate"
            )
            return
        discarded = self.draft.dirty
        self.index = next_index
        if discarded:
            notice = f"{notice}; unsaved changes discarded"
        self._load_current(notice)

    def _request_quit(self) -> None:
        if self.draft.dirty:
            self.mode.previous = self.mode.name
            self.mode.name = "quit"
            return
        self._running = False

    def _handle_quit_confirmation(self, key: object) -> None:
        if isinstance(key, str) and key.lower() == "y":
            self._running = False
        elif self._is_escape(key) or (isinstance(key, str) and key.lower() == "n"):
            self.mode.name = self.mode.previous

    def _set_notice(self, text: str, *, error: bool = False) -> None:
        self.notice = str(text)
        self.notice_error = bool(error)
        self._image_gate_notice = False

    def _phase_search_bounds(self, payload: dict | None) -> tuple[float, float]:
        if self._phase_search_window_index == 0:
            from malca.review.period_search import adaptive_review_period_bounds

            return adaptive_review_period_bounds(payload)
        return self._phase_search_windows[self._phase_search_window_index]

    def _phase_search_window_label(self) -> str:
        payload = getattr(self.record, "payload", None)
        payload_dict = dict(payload) if isinstance(payload, dict) else None
        lo, hi = self._phase_search_bounds(payload_dict)
        if self._phase_search_window_index == 0:
            return f"adaptive {lo:g}–{hi:g} d"
        return f"{lo:g}–{hi:g} d"

    def _time_window_label(self) -> str:
        if self.time_window_mode != "asassn":
            return "full baseline"
        padding = float(self.asassn_window_padding_days)
        return (
            f"ASAS-SN span ±{padding:g} d"
            if padding > 0.0
            else "ASAS-SN span"
        )

    def _external_photometry_status(self, *, compact: bool) -> str:
        """Describe the master external toggle and selected source set."""

        if not self.show_external_lightcurves:
            return "ext−" if compact else "external off"
        if not self.external_lightcurve_sources:
            return "ext none" if compact else "external on · ASAS-SN only"

        if compact:
            short_labels = {
                "atlas": "ATLAS",
                "ztf": "ZTF",
                "gaia_epoch": "Gaia",
                "neowise": "WISE",
                "allwise_mep": "AllWISE",
                "aavso": "AAVSO",
                "ogle": "OGLE",
                "stripe82": "S82",
                "vvvx_virac": "VVVX",
                "ps1": "PS1",
                "superwasp": "SWASP",
                "kelt": "KELT",
                "nsvs": "NSVS",
                "asas3": "ASAS3",
                "crts": "CRTS",
                "dasch": "DASCH",
            }
            labels = [
                short_labels.get(source, source.upper())
                for source in self.external_lightcurve_sources
            ]
            visible = labels[:3]
            suffix = f"+{len(labels) - 3}" if len(labels) > 3 else ""
            return "ext " + "+".join(visible) + suffix

        labels = [
            tui_external_photometry_source_label(source)
            for source in self.external_lightcurve_sources
        ]
        return "external " + ", ".join(labels)

    def _export_current_render(self) -> None:
        """Copy the current review PNG next to the review DB.

        Fast-triage exports are the plot exactly as the user is looking at
        it, so we never re-render; we just clone the coordinator's canonical
        image.  Missing renders surface as an error notice instead of raising.
        """
        from shutil import copy2

        candidate_id = self.candidate_id
        if not candidate_id:
            self._set_notice("Nothing to export", error=True)
            return
        source_path = getattr(self.images.status, "path", None)
        if source_path is None or not Path(source_path).is_file():
            self._set_notice("No rendered image yet", error=True)
            return
        export_dir = self.db_path.parent / "tui-exports"
        try:
            export_dir.mkdir(parents=True, exist_ok=True)
            destination = export_dir / f"{candidate_id}.png"
            copy2(source_path, destination)
        except OSError as exc:
            self._set_notice(f"Export failed: {exc}", error=True)
            return
        self._set_notice(f"Exported {destination}")

    def _image_is_ready(self) -> bool:
        return self.image_status == "ready"

    def _block_for_image(self) -> None:
        if self.image_status.startswith("error:"):
            self.notice = "Current image failed; Tab skips this candidate"
        else:
            self.notice = "Wait for the current image; Tab skips this candidate"
        self.notice_error = True
        self._image_gate_notice = True

    def _is_candidate_edit_key(self, key: object) -> bool:
        if isinstance(key, str) and key in {"1", "2", "3", "4", ",", "."}:
            return True
        if self._is_enter(key):
            return True
        if self.mode.name == "subtypes":
            if self._is_backspace(key):
                return True
            if not isinstance(key, str) or not self.draft.morphology_primary:
                return False
            return key.lower() in MORPHOLOGY_SECONDARY_BY_KEY.get(
                self.draft.morphology_primary, {}
            )
        if self.mode.name == "physical":
            if self._is_backspace(key):
                return True
            return isinstance(key, str) and key.lower() in PHYSICAL_PRIMARY_BY_KEY
        if self.mode.name == "physical_subtypes":
            if self._is_backspace(key):
                return True
            if not isinstance(key, str) or not self.draft.physical_primary:
                return False
            return key.lower() in PHYSICAL_SECONDARY_BY_KEY.get(
                self.draft.physical_primary, {}
            )
        return isinstance(key, str) and key.lower() in MORPHOLOGY_PRIMARY_BY_KEY

    def _persist_position_best_effort(self) -> None:
        candidate_id = self.candidate_id
        if candidate_id is None:
            return
        try:
            self.repository.persist_last_candidate(candidate_id)
        except Exception:
            # Resume state is a convenience.  A busy app_state table must not
            # prevent review navigation or turn a clean quit into a crash.
            pass

    def _draw(self) -> None:
        self.screen.erase()
        height, width = self.screen.getmaxyx()
        if height < MIN_TERMINAL_HEIGHT or width < MIN_TERMINAL_WIDTH:
            self._add(
                0,
                0,
                f"Terminal too small: {width}x{height}",
                self._styles["error"],
            )
            self._add(
                1,
                0,
                f"Minimum: {MIN_TERMINAL_WIDTH}x{MIN_TERMINAL_HEIGHT}",
            )
            if self.mode.name == "quit":
                self._add(3, 0, "Dirty draft: y quit  n return")
            elif self.mode.name in {
                "filters",
                "filter_confirm",
                "filter_morphology",
                "filter_physical",
                "filter_catalog_types",
            }:
                self._add(3, 0, "Resize terminal; Esc returns")
            else:
                self._add(3, 0, "Resize terminal or press Q to quit")
            self.screen.refresh()
            return
        if self.mode.name == "help":
            self._draw_help()
        elif self.mode.name == "quit":
            self._draw_main()
            self._draw_centered_prompt("Discard unsaved changes and quit? [y/N]")
        elif self.mode.name in {"filters", "filter_confirm"}:
            self._draw_filters()
            if self.mode.name == "filter_confirm":
                self._draw_centered_prompt(
                    "Dirty: [s] save  [d] discard  [Esc] cancel"
                )
        elif self.mode.name in {"filter_morphology", "filter_physical"}:
            self._draw_filter_taxonomy()
        elif self.mode.name == "filter_catalog_types":
            self._draw_catalog_type_filters()
        else:
            self._draw_main()
            if self.mode.name == "search_input":
                self._draw_centered_prompt(f"Candidate / ASAS-SN ID: {self.input_buffer}_")
            elif self.mode.name == "period_input":
                self._draw_centered_prompt(f"Display period (days): {self.input_buffer}_")
            elif self.mode.name == "notes_input":
                visible_notes = self.input_buffer.replace("\n", " ↵ ")
                self._draw_centered_prompt(f"Review notes: {visible_notes}_")
            elif self.mode.name == "search_confirm":
                self._draw_centered_prompt(
                    "Dirty: [s] save  [d] discard  [Esc] cancel"
                )
        self.screen.refresh()

    def _draw_main(self) -> None:
        # Width changes reflow this one complete content set; they must not
        # select a different view that reveals or omits review evidence.
        self._draw_main_portrait()

    def _draw_main_portrait(self) -> None:
        """Draw a stable vertical stack for narrow, tall terminals."""

        height, width = self.screen.getmaxyx()
        position = (
            f"{self.index + 1}/{len(self.candidate_ids)}"
            if self.candidate_ids
            else "0/0"
        )
        active_filter = getattr(self.repository, "filter_spec", self.filter_editor.spec)
        filter_parts = list(active_filter.summary_parts())
        if bool(getattr(self.repository, "search_override", False)):
            filter_parts.insert(0, "search override")
        title = "MALCA REVIEW"
        self._add(0, 0, title, self._styles["title"])
        self._add(0, max(0, width - len(position) - 1), position, self._styles["label"])
        filter_line = " · ".join(filter_parts) or "—"
        identity_row = self._draw_wrapped_text(
            1,
            0,
            filter_line,
            max_width=width - 1,
            style=self._styles["dim"],
        )

        candidate = self.record
        asas_sn_id = candidate.asas_sn_id if candidate else None
        payload = getattr(candidate, "payload", None)
        candidate_line = build_review_identity_line(
            payload if isinstance(payload, dict) else None,
            asas_sn_id=asas_sn_id,
        )
        catalog_row = self._draw_wrapped_text(
            identity_row,
            0,
            candidate_line,
            max_width=width - 1,
            style=self._styles["value"],
        )
        metrics_row = self._draw_catalog_line(catalog_row, payload, width - 1)

        metrics_row = self._draw_fold_view_context(
            metrics_row,
            max_width=width - 1,
        )

        primary_label = label_for(self.draft.morphology_primary) or "unselected"
        physical_label = label_for(self.draft.physical_primary) or "unselected"
        physical_secondary_label = label_for(self.draft.physical_secondary) or "none"
        secondary_labels = [label_for(value) for value in self.draft.morphology_secondaries]

        detail_values = _detail_value_map(
            detail_sections(
                getattr(candidate, "payload", None),
                phase_period_days=self.phase_period_days,
                phase_period_source=self.phase_source,
                width=10_000,
            )
        )
        row = self._draw_triage_detail_sections(
            metrics_row,
            detail_values,
            max_width=width - 1,
        )
        row = self._draw_compact_draft_section(
            row,
            primary_label=primary_label,
            secondary_labels=secondary_labels,
            physical_label=physical_label,
            physical_secondary_label=physical_secondary_label,
        )

        footer_start = height - PORTRAIT_FOOTER_HEIGHT
        self._draw_active_menu(
            header_y=row,
            start_y=row + 1,
            end_y=footer_start - 1,
            primary_label=primary_label,
        )
        self._draw_footer(footer_start)

    def _draw_main_landscape(self) -> None:
        height, width = self.screen.getmaxyx()
        position = (
            f"{self.index + 1} / {len(self.candidate_ids)}"
            if self.candidate_ids
            else "0 / 0"
        )
        mode_label = {
            "subtypes": "SUBTYPES",
            "physical": "PHYSICAL",
            "physical_subtypes": "PHYS SUBTYPE",
        }.get(self.mode.name, "PRIMARY")
        active_filter = getattr(self.repository, "filter_spec", self.filter_editor.spec)
        summary = active_filter.summary()
        if bool(getattr(self.repository, "search_override", False)):
            summary = f"search override · {summary}"
        right_header = f"{mode_label}   {position}"
        title_room = max(12, width - len(right_header) - 2)
        self._add(
            0,
            0,
            f"MALCA REVIEW  [{summary}]"[:title_room],
            self._styles["title"],
        )
        self._add(0, max(0, width - len(right_header) - 1), right_header)

        candidate = self.record
        asas_sn_id = candidate.asas_sn_id if candidate else None
        payload = getattr(candidate, "payload", None)
        candidate_line = build_review_identity_line(
            payload if isinstance(payload, dict) else None,
            asas_sn_id=asas_sn_id,
        )
        self._add(1, 0, candidate_line, self._styles["value"])
        phase_text = _format_period(self.phase_period_days)
        if self.phase_source:
            phase_text += f" ({self.phase_source})"
        marker_text = "events+" if self.show_event_markers else "events−"
        detail_bits = [
            f"Phase {phase_text}",
            f"PDM {self._phase_search_window_label()}",
            self.camera_view,
            "colors+" if self.color_by_camera else "colors−",
            self._external_photometry_status(compact=True),
            "span ASAS-SN"
            if self.time_window_mode == "asassn"
            else "span full",
            marker_text,
        ]
        self._add(
            2,
            0,
            " · ".join(detail_bits),
            self._styles["dim"],
        )

        detail_x = max(48, width // 2)
        left_width = detail_x - 2
        primary_label = label_for(self.draft.morphology_primary) or "unselected"
        physical_label = label_for(self.draft.physical_primary) or "unselected"
        physical_secondary_label = label_for(self.draft.physical_secondary) or "none"
        self._add_clipped(3, 0, f"Morphology: {primary_label}", left_width)
        secondary_labels = [label_for(value) for value in self.draft.morphology_secondaries]
        self._add_clipped(
            4,
            0,
            "Subtypes: " + (", ".join(secondary_labels) or "none"),
            left_width,
        )
        self._add_clipped(5, 0, f"Physical: {physical_label}", left_width)
        self._add_clipped(
            6, 0, f"Physical subtype: {physical_secondary_label}", left_width
        )
        confidence = self.draft.confidence if self.draft.confidence is not None else "unset"
        self._add_clipped(
            7,
            0,
            f"Confidence: {confidence}    Follow-up: "
            f"{'yes' if self.draft.needs_followup else 'no'}",
            left_width,
        )
        notes_preview = _ellipsize_text(self.draft.notes.replace("\n", " ↵ ") or "none", left_width - 7)
        self._add_clipped(8, 0, f"Notes: {notes_preview}", left_width)
        workflow = str(self.review.get("workflow_status") or "unreviewed")
        state = "modified, not saved" if self.draft.dirty else workflow
        state_style = self._styles["dirty"] if self.draft.dirty else self._styles["dim"]
        self._add_clipped(9, 0, f"State: {state}", left_width, state_style)

        detail_width = width - detail_x - 1
        detail_values = _detail_value_map(
            detail_sections(
                getattr(candidate, "payload", None),
                phase_period_days=self.phase_period_days,
                phase_period_source=self.phase_source,
                width=detail_width,
            )
        )
        vetting_end_row = self._draw_signal_and_vetting_sections(
            3,
            detail_values,
            column=detail_x,
            max_width=detail_width,
        )
        # Draw catalog labels with coloured values instead of a flat text run
        # so ``VSX SRS`` reads as ``VSX <value>`` at a glance.
        col = detail_x
        row = vetting_end_row
        label_style = self._styles["label"]
        value_style = self._styles["catalog"]

        def emit_cat(label: str, value: str, *, attr: int | None = None) -> None:
            nonlocal col
            if not value or value == "—":
                return
            if col > detail_x:
                self._add(row, col, "  ", label_style)
                col += 2
            self._add(row, col, f"{label} ", label_style)
            col += len(label) + 1
            self._add(row, col, value, attr if attr is not None else value_style)
            col += len(value)

        emit_cat("VSX", str(detail_values.get("VSX", "—")))
        emit_cat("Gaia", str(detail_values.get("Gaia VAR", "—")))
        known_value = str(detail_values.get("known", "—"))
        emit_cat(
            "Known",
            known_value,
            attr=self._styles["dirty"] if known_value == "yes" else value_style,
        )

        body_start = max(12, vetting_end_row + 1)
        footer_height = 5
        body_end = height - footer_height - 1
        if self.mode.name == "subtypes":
            primary = self.draft.morphology_primary
            items = MORPHOLOGY_SECONDARY_ITEMS.get(primary or "", ())
            self._add(body_start, 0, f"{primary_label.upper()} SUBTYPES — letter keys toggle", self._styles["section"])
            self._draw_menu(
                items,
                selected=set(self.draft.morphology_secondaries),
                start_y=body_start + 1,
                end_y=body_end,
            )
        elif self.mode.name == "physical":
            self._add(
                body_start,
                0,
                "BROAD PHYSICAL HYPOTHESIS — one optional label",
                self._styles["section"],
            )
            self._draw_menu(
                PHYSICAL_PRIMARY_ITEMS,
                selected={self.draft.physical_primary} if self.draft.physical_primary else set(),
                start_y=body_start + 1,
                end_y=body_end,
            )
        elif self.mode.name == "physical_subtypes":
            items = PHYSICAL_SECONDARY_ITEMS.get(
                self.draft.physical_primary or "", ()
            )
            self._add(
                body_start,
                0,
                f"{physical_label.upper()} SUBTYPES — one optional label",
                self._styles["section"],
            )
            self._draw_menu(
                items,
                selected=(
                    {self.draft.physical_secondary}
                    if self.draft.physical_secondary else set()
                ),
                start_y=body_start + 1,
                end_y=body_end,
            )
        else:
            self._add(body_start, 0, "PRIMARY MORPHOLOGY — selecting one opens its subtypes", self._styles["section"])
            self._draw_menu(
                MORPHOLOGY_PRIMARY_ITEMS,
                selected={self.draft.morphology_primary} if self.draft.morphology_primary else set(),
                start_y=body_start + 1,
                end_y=body_end,
            )
        self._draw_footer(height - footer_height)

    def _draw_active_menu(
        self,
        *,
        header_y: int,
        start_y: int,
        end_y: int,
        primary_label: str,
    ) -> None:
        """Draw whichever taxonomy menu is active below a section title."""

        if self.mode.name == "subtypes":
            primary = self.draft.morphology_primary
            self._draw_section_header(header_y, f"{primary_label.upper()} SUBTYPES")
            self._draw_menu(
                MORPHOLOGY_SECONDARY_ITEMS.get(primary or "", ()),
                selected=set(self.draft.morphology_secondaries),
                start_y=start_y,
                end_y=end_y,
            )
        elif self.mode.name == "physical":
            self._draw_section_header(header_y, "PHYSICAL HYPOTHESIS")
            self._draw_menu(
                PHYSICAL_PRIMARY_ITEMS,
                selected={self.draft.physical_primary} if self.draft.physical_primary else set(),
                start_y=start_y,
                end_y=end_y,
            )
        elif self.mode.name == "physical_subtypes":
            physical_label = label_for(self.draft.physical_primary) or "PHYSICAL"
            self._draw_section_header(
                header_y, f"{physical_label.upper()} SUBTYPES"
            )
            self._draw_menu(
                PHYSICAL_SECONDARY_ITEMS.get(
                    self.draft.physical_primary or "", ()
                ),
                selected=(
                    {self.draft.physical_secondary}
                    if self.draft.physical_secondary else set()
                ),
                start_y=start_y,
                end_y=end_y,
            )
        else:
            self._draw_section_header(header_y, "PRIMARY MORPHOLOGY")
            self._draw_menu(
                MORPHOLOGY_PRIMARY_ITEMS,
                selected={self.draft.morphology_primary} if self.draft.morphology_primary else set(),
                start_y=start_y,
                end_y=end_y,
            )

    def _draw_section_header(
        self,
        row: int,
        title: str,
        *,
        column: int = 0,
        max_width: int | None = None,
    ) -> None:
        """Draw a compact section title followed by a light horizontal rule."""

        _height, terminal_width = self.screen.getmaxyx()
        available = (
            terminal_width - column - 1 if max_width is None else max(0, int(max_width))
        )
        if available <= 0:
            return
        clipped_title = str(title)[:available]
        self._add(row, column, clipped_title, self._styles["title"])
        rule_start = column + len(clipped_title) + 1
        rule_length = max(0, available - len(clipped_title) - 1)
        if rule_length:
            self._add(row, rule_start, "─" * rule_length, self._styles["dim"])

    def _draw_catalog_line(self, row: int, payload: object, max_width: int) -> int:
        """Render catalog labels, flowing complete entries onto later rows.

        A catalog class is evidence, not decorative metadata: squeezing VSX
        and Gaia onto the first row must never cause SIMBAD (or a later PM /
        known flag) to vanish.  Entries therefore move to the next row as a
        unit, and an exceptionally long value is wrapped rather than clipped.
        """

        _height, terminal_width = self.screen.getmaxyx()
        available = min(int(max_width), terminal_width - 1)
        if available <= 0:
            return row
        detail_values = _detail_value_map(
            detail_sections(
                payload if isinstance(payload, dict) else None,
                phase_period_days=None,
                phase_period_source=None,
                # Catalog evidence is flowed below.  Do not let the generic
                # detail formatter ellipsize SIMBAD before this renderer has
                # a chance to wrap the complete value.
                width=10_000,
            )
        )

        col = 0
        label_style = self._styles["label"]
        value_style = self._styles["catalog"]

        def emit(label: str, value: str, *, value_attr: int | None = None) -> None:
            nonlocal row, col
            prefix = f"{label} "
            token_width = len(prefix) + len(value)
            if col and col + 2 + token_width > available:
                row += 1
                col = 0
            if col:
                self._add(row, col, "  ", label_style)
                col += 2
            self._add(row, col, prefix, label_style)
            col += len(prefix)
            style = value_attr if value_attr is not None else value_style
            value_chunks = textwrap.wrap(
                value,
                width=max(1, available - col),
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
            self._add(row, col, value_chunks[0], style)
            col += len(value_chunks[0])
            for chunk in value_chunks[1:]:
                row += 1
                col = 0
                self._add(row, col, chunk, style)
                col = len(chunk)

        for label, key in (
            ("VSX", "VSX"),
            ("Gaia", "Gaia VAR"),
            ("ASAS-SN", "ASAS-SN"),
            ("SIMBAD", "SIMBAD"),
        ):
            value = detail_values.get(key, "—")
            if value and value != "—":
                emit(label, str(value))

        pm_value = detail_values.get("PM", "—")
        if pm_value and pm_value != "—":
            emit(
                "PM",
                str(pm_value),
                value_attr=self._pm_field_style(pm_value),
            )

        known_value = detail_values.get("known", "—")
        if known_value and known_value != "—":
            attr = self._styles["dirty"] if known_value == "yes" else value_style
            emit("Known", str(known_value), value_attr=attr)
        return row + 1

    def _draw_fold_view_context(self, start_row: int, *, max_width: int) -> int:
        """Draw aligned fold and view rows matching the metrics blocks."""

        row = start_row
        fold_bits = [f"P {_format_period(self.phase_period_days)}"]
        source = _short_phase_source(self.phase_source)
        if source:
            fold_bits.append(source)
        fold_bits.append(self._phase_search_window_label())
        row = self._draw_wrapped_field(
            row,
            "fold",
            " · ".join(fold_bits),
            max_width=max_width,
        )
        view_bits = [
            f"cam {self.camera_view}",
            "colors on" if self.color_by_camera else "colors off",
            self._external_photometry_status(compact=False),
            f"window {self._time_window_label()}",
            "events on" if self.show_event_markers else "events off",
            f"theme {self.plot_theme}",
        ]
        row = self._draw_wrapped_field(
            row,
            "view",
            " · ".join(view_bits),
            max_width=max_width,
        )
        return row

    def _draw_field(
        self,
        row: int,
        label: str,
        value: object,
        *,
        column: int = 0,
        max_width: int | None = None,
        value_style: int | None = None,
    ) -> None:
        """Draw an aligned label/value pair in the review sidebar."""

        _height, terminal_width = self.screen.getmaxyx()
        available = (
            terminal_width - column - 1 if max_width is None else max(0, int(max_width))
        )
        if available <= 0:
            return
        label_text = f"  {label:<11}"
        if len(label_text) > available:
            label_text = label_text[:available]
        self._add(row, column, label_text, self._styles["label"])
        value_column = column + len(label_text)
        self._add_clipped(
            row,
            value_column,
            value,
            available - len(label_text),
            self._styles["value"] if value_style is None else value_style,
        )

    def _draw_wrapped_text(
        self,
        row: int,
        column: int,
        text: object,
        *,
        max_width: int,
        style: int = 0,
    ) -> int:
        """Draw all text, returning the row immediately after its wrapping."""

        _height, terminal_width = self.screen.getmaxyx()
        available = min(max(0, int(max_width)), terminal_width - column - 1)
        if available <= 0:
            return row
        lines = textwrap.wrap(
            str(text),
            width=available,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        for offset, line in enumerate(lines):
            self._add(row + offset, column, line, style)
        return row + len(lines)

    def _draw_wrapped_field(
        self,
        row: int,
        label: str,
        value: object,
        *,
        column: int = 0,
        max_width: int | None = None,
        value_style: int | None = None,
    ) -> int:
        """Draw a labelled value without discarding text at a narrow width."""

        _height, terminal_width = self.screen.getmaxyx()
        available = (
            terminal_width - column - 1 if max_width is None else max(0, int(max_width))
        )
        if available <= 0:
            return row
        label_text = f"  {label:<11}"
        if len(label_text) >= available:
            return self._draw_wrapped_text(
                row,
                column,
                f"{label}: {value}",
                max_width=available,
                style=self._styles["value"] if value_style is None else value_style,
            )
        value_column = column + len(label_text)
        value_width = available - len(label_text)
        value_lines = textwrap.wrap(
            str(value),
            width=max(1, value_width),
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        self._add(row, column, label_text, self._styles["label"])
        style = self._styles["value"] if value_style is None else value_style
        self._add(row, value_column, value_lines[0], style)
        continuation = " " * len(label_text)
        for offset, line in enumerate(value_lines[1:], start=1):
            self._add(row + offset, column, continuation, self._styles["label"])
            self._add(row + offset, value_column, line, style)
        return row + len(value_lines)

    def _draw_field_pair(
        self,
        row: int,
        left: tuple[str, object],
        right: tuple[str, object],
        *,
        column: int = 0,
        max_width: int | None = None,
    ) -> None:
        """Draw two compact label/value pairs on one row when space allows."""

        _height, terminal_width = self.screen.getmaxyx()
        available = (
            terminal_width - column - 1 if max_width is None else max(0, int(max_width))
        )
        if available <= 0:
            return
        left_label, left_value = left
        right_label, right_value = right
        split = column + max(20, available // 2)
        self._draw_field(
            row,
            left_label,
            left_value,
            column=column,
            max_width=max(0, split - column),
        )
        if split < column + available - 12:
            self._draw_field(
                row,
                right_label,
                right_value,
                column=split,
                max_width=available - (split - column),
            )

    def _draw_ml_class_score_rows(
        self,
        row: int,
        values: Mapping[str, str],
        *,
        column: int = 0,
        max_width: int | None = None,
    ) -> int:
        """Render the hierarchical gate, primary, and subtype ML scores."""

        _height, terminal_width = self.screen.getmaxyx()
        available = (
            terminal_width - column - 1 if max_width is None else max(0, int(max_width))
        )
        if available <= 0:
            return 0
        for row_offset, score_group in enumerate(
            (
                ML_CLASS_SCORE_FIELDS[:2],
                ML_CLASS_SCORE_FIELDS[2:5],
                ML_CLASS_SCORE_FIELDS[5:7],
                ML_CLASS_SCORE_FIELDS[7:10],
                ML_CLASS_SCORE_FIELDS[10:],
            )
        ):
            col = column + 2
            for index, (_column, detail_label, short_label) in enumerate(score_group):
                if index:
                    col += 1
                token = f"{short_label} "
                if col + len(token) > column + available:
                    break
                self._add(row + row_offset, col, token, self._styles["label"])
                col += len(token)
                value = values.get(detail_label, "—")
                style = self._styles["value"]
                try:
                    if float(value) >= 0.5:
                        style = self._styles["dirty"]
                except (TypeError, ValueError):
                    pass
                if col + len(value) > column + available:
                    value = value[: max(0, column + available - col)]
                self._add(row + row_offset, col, value, style)
                col += len(value) + 2
        return 5

    def _draw_signal_metrics_row(
        self,
        row: int,
        values: Mapping[str, str],
        *,
        column: int = 0,
        max_width: int | None = None,
    ) -> None:
        """Render the non-ML Q/M morphology diagnostics on one scannable row."""

        _height, terminal_width = self.screen.getmaxyx()
        available = (
            terminal_width - column - 1 if max_width is None else max(0, int(max_width))
        )
        if available <= 0:
            return
        col = column + 2
        for index, key in enumerate(("Q", "M")):
            if index:
                col += 1
            token = f"{key} "
            if col + len(token) > column + available:
                break
            self._add(row, col, token, self._styles["label"])
            col += len(token)
            value = values.get(key, "—")
            if col + len(value) > column + available:
                value = value[: max(0, column + available - col)]
            self._add(row, col, value, self._styles["value"])
            col += len(value) + 2

    def _draw_triage_detail_sections(
        self,
        start_row: int,
        detail_values: Mapping[str, str],
        *,
        column: int = 0,
        max_width: int | None = None,
    ) -> int:
        """Draw candidate metrics grouped for fast dipper triage."""

        row = start_row
        self._draw_section_header(row, "ML CLASS SCORES", column=column, max_width=max_width)
        row += 1
        row += self._draw_ml_class_score_rows(
            row,
            detail_values,
            column=column,
            max_width=max_width,
        )
        self._draw_signal_metrics_row(
            row,
            detail_values,
            column=column,
            max_width=max_width,
        )
        row += 1
        row = self._draw_wrapped_field(
            row,
            "mean mag",
            detail_values.get("mean mag", "—"),
            column=column,
            max_width=max_width,
        )

        sed_bits: list[str] = []
        alpha_value = detail_values.get("α_SED", "—")
        alpha_class = detail_values.get("α class", "—")
        if not _is_missing_detail_value(alpha_value):
            sed_bits.append(f"α {alpha_value}")
        if not _is_missing_detail_value(alpha_class):
            sed_bits.append(str(alpha_class))
        if sed_bits:
            self._draw_section_header(row, "SED", column=column, max_width=max_width)
            row += 1
            row = self._draw_wrapped_field(
                row,
                "shape",
                " · ".join(sed_bits),
                column=column,
                max_width=max_width,
            )

        star_rows = [
            ("Teff", detail_values.get("Teff", "—")),
            ("type", detail_values.get("type", "—")),
            ("log g", detail_values.get("log g", "—")),
            ("[M/H]", detail_values.get("[M/H]", "—")),
            ("Mass", detail_values.get("Mass", "—")),
            ("A_V", detail_values.get("A_V", "—")),
        ]
        star_rows = [
            (label, value) for label, value in star_rows if not _is_missing_detail_value(value)
        ]
        if star_rows:
            self._draw_section_header(row, "STAR", column=column, max_width=max_width)
            row += 1
            index = 0
            while index < len(star_rows):
                left = star_rows[index]
                right = star_rows[index + 1] if index + 1 < len(star_rows) else None
                if right is None:
                    row = self._draw_wrapped_field(
                        row,
                        left[0],
                        left[1],
                        column=column,
                        max_width=max_width,
                    )
                elif (max_width or 0) < 72:
                    row = self._draw_wrapped_field(
                        row,
                        left[0],
                        left[1],
                        column=column,
                        max_width=max_width,
                    )
                    row = self._draw_wrapped_field(
                        row,
                        right[0],
                        right[1],
                        column=column,
                        max_width=max_width,
                    )
                else:
                    self._draw_field_pair(
                        row,
                        left,
                        right,
                        column=column,
                        max_width=max_width,
                    )
                    row += 1
                index += 2

        flag_rows = [
            ("RUWE", detail_values.get("RUWE", "—"), None),
            ("PM", detail_values.get("PM", "—"), "pm"),
            ("EB", detail_values.get("EB", "—"), None),
            ("BANYAN", detail_values.get("BANYAN", "—"), None),
        ]
        flag_rows = [
            (label, value, style_key)
            for label, value, style_key in flag_rows
            if not _is_missing_detail_value(value)
        ]
        if flag_rows:
            self._draw_section_header(row, "FLAGS", column=column, max_width=max_width)
            row += 1
            index = 0
            while index < len(flag_rows):
                left = flag_rows[index]
                right = flag_rows[index + 1] if index + 1 < len(flag_rows) else None
                if right is None:
                    row = self._draw_wrapped_field(
                        row,
                        left[0],
                        left[1],
                        column=column,
                        max_width=max_width,
                        value_style=self._flag_value_style(left[1], left[2]),
                    )
                elif (max_width or 0) < 72:
                    row = self._draw_wrapped_field(
                        row,
                        left[0],
                        left[1],
                        column=column,
                        max_width=max_width,
                        value_style=self._flag_value_style(left[1], left[2]),
                    )
                    row = self._draw_wrapped_field(
                        row,
                        right[0],
                        right[1],
                        column=column,
                        max_width=max_width,
                        value_style=self._flag_value_style(right[1], right[2]),
                    )
                else:
                    split = column + max(20, (max_width or 1) // 2)
                    self._draw_field(
                        row,
                        left[0],
                        left[1],
                        column=column,
                        max_width=max(0, split - column),
                        value_style=self._flag_value_style(left[1], left[2]),
                    )
                    self._draw_field(
                        row,
                        right[0],
                        right[1],
                        column=split,
                        max_width=(max_width or 1) - (split - column),
                        value_style=self._flag_value_style(right[1], right[2]),
                    )
                    row += 1
                index += 2
        return row

    def _pm_field_style(self, value: object) -> int:
        """Highlight total PM at/above the pipeline high-PM threshold."""

        try:
            pm = float(str(value).strip())
        except (TypeError, ValueError):
            return self._styles["value"]
        if pm >= float(POST_FILTER_MAX_PM):
            return self._styles["dirty"]
        return self._styles["value"]

    def _flag_value_style(self, value: object, style_key: str | None) -> int:
        if style_key == "pm":
            return self._pm_field_style(value)
        return self._styles["value"]

    def _draw_compact_draft_section(
        self,
        start_row: int,
        *,
        primary_label: str,
        secondary_labels: list[str],
        physical_label: str,
        physical_secondary_label: str,
    ) -> int:
        """Summarize selected in-progress review labels."""

        row = start_row
        draft_bits: list[str] = []
        if primary_label != "unselected":
            draft_bits.append(primary_label)
        if secondary_labels:
            draft_bits.append(", ".join(secondary_labels))
        if physical_label != "unselected":
            draft_bits.append(physical_label)
        if physical_secondary_label != "none":
            draft_bits.append(physical_secondary_label)
        self._draw_section_header(row, "DRAFT", column=0, max_width=None)
        row += 1
        row = self._draw_wrapped_field(
            row,
            "labels",
            " · ".join(draft_bits) if draft_bits else "—",
        )

        return row

    def _draw_signal_and_vetting_sections(
        self,
        start_row: int,
        detail_values: Mapping[str, str],
        *,
        column: int = 0,
        max_width: int | None = None,
    ) -> int:
        """Landscape/detail-column alias for the triage metric blocks."""

        return self._draw_triage_detail_sections(
            start_row,
            detail_values,
            column=column,
            max_width=max_width,
        )

    def _draw_filters(self) -> None:
        height, width = self.screen.getmaxyx()
        count = "?" if self.filter_match_count < 0 else str(self.filter_match_count)
        count_text = f"{count} matches"
        self._add(0, 0, "FILTERS", self._styles["title"])
        self._add(0, max(10, width - len(count_text) - 1), count_text, self._styles["label"])
        self._add(
            1,
            0,
            _compact_token_line(self.filter_editor.spec.summary_parts(), width - 1),
            self._styles["dim"],
        )

        rows = self.filter_editor.rows()
        start_y = 3
        end_y = height - 5
        visible = max(1, end_y - start_y + 1)
        scroll = min(
            max(0, self.filter_editor.cursor - visible + 1),
            max(0, len(rows) - visible),
        )
        label_width = min(
            19 if width <= PORTRAIT_MAX_WIDTH else 24,
            max(len(row.label) for row in rows) + 2,
        )
        for row_index, row in enumerate(rows[scroll:scroll + visible], start=scroll):
            if row.kind == "heading":
                self._add(
                    start_y + row_index - scroll,
                    0,
                    f"  {row.label}",
                    self._styles["label"],
                )
                continue
            selected = row_index == self.filter_editor.cursor
            marker = ">" if selected else " "
            text = f"{marker} {row.label + ':':<{label_width}} {row.value}"
            if row.kind in {"taxonomy", "catalog"} and width > PORTRAIT_MAX_WIDTH:
                text += "   [Space to choose]"
            self._add(
                start_y + row_index - scroll,
                0,
                text,
                self._styles["selected"] if selected else 0,
            )
        active_kind = rows[self.filter_editor.cursor].kind
        if width <= PORTRAIT_MAX_WIDTH:
            context = (
                "Space opens choices  ↑/↓ row"
                if active_kind in {"taxonomy", "catalog"}
                else "←/→ change  Space cycle  ↑/↓ row"
            )
            self._add(height - 3, 0, context)
            self._add(height - 2, 0, "Enter apply  R reset  Esc cancel")
        else:
            self._add(height - 3, 0, "[↑/↓] row  [←/→] change  [Space] change/open")
            self._add(height - 2, 0, "[Enter] apply   [R] reset   [Esc] cancel")
        notice_style = self._styles["error"] if self.notice_error else self._styles["dim"]
        self._add(height - 1, 0, self.notice, notice_style)

    def _draw_filter_taxonomy(self) -> None:
        height, width = self.screen.getmaxyx()
        morphology = self.mode.name == "filter_morphology"
        title = "FILTER MORPHOLOGY" if morphology else "FILTER PHYSICAL HYPOTHESIS"
        count = "?" if self.filter_match_count < 0 else str(self.filter_match_count)
        self._add(0, 0, title, self._styles["title"])
        self._add(0, max(10, width - len(str(count)) - 10), f"{count} matches")
        instruction = (
            "Letters toggle · ALL/ANY is set in Filters"
            if width <= PORTRAIT_MAX_WIDTH
            else "Letter keys toggle; selected values use ALL/ANY from filter screen"
        )
        self._add(1, 0, instruction)
        items = MORPHOLOGY_PRIMARY_ITEMS if morphology else PHYSICAL_PRIMARY_ITEMS
        selected = set(
            self.filter_editor.spec.morphology_primary
            if morphology
            else self.filter_editor.spec.physical_primary
        )
        self._draw_menu(items, selected=selected, start_y=3, end_y=height - 4)
        footer = (
            "Esc filters  Backspace clear  Pg keys pages"
            if width <= PORTRAIT_MAX_WIDTH
            else "[Esc] filters  [Backspace] clear  [PgUp/PgDn] pages"
        )
        self._add(height - 2, 0, footer)
        notice_style = self._styles["error"] if self.notice_error else self._styles["dim"]
        self._add(height - 1, 0, self.notice, notice_style)

    def _draw_catalog_type_filters(self) -> None:
        """Draw one campaign-local catalog's keep/exclude menu."""

        height, width = self.screen.getmaxyx()
        count = "?" if self.filter_match_count < 0 else str(self.filter_match_count)
        stats = self._active_catalog_type_stats()
        catalog_label = (
            stats[0].catalog_label
            if stats
            else (self.filter_editor.active_catalog or "Catalog").upper()
        )
        self._add(0, 0, f"{catalog_label} TYPES", self._styles["title"])
        self._add(
            0,
            max(10, width - len(count) - 10),
            f"{count} matches",
            self._styles["label"],
        )
        self._add(
            1,
            0,
            "Y keep · N exclude · Space toggle · default YES",
        )

        if not stats:
            self._add(3, 0, f"No {catalog_label} types are stored.")
            self._add(height - 2, 0, "Enter/Esc returns to Filters")
            notice_style = (
                self._styles["error"] if self.notice_error else self._styles["dim"]
            )
            self._add(height - 1, 0, self.notice, notice_style)
            return

        total_candidates = max(
            int(getattr(stat, "total_candidates", 0)) for stat in stats
        )
        self._add(
            2,
            0,
            f"{total_candidates:,} campaign candidates · {len(stats)} stored types",
            self._styles["dim"],
        )

        start_y = 4
        end_y = height - 4
        rows_per_item = 3
        visible_items = max(1, (end_y - start_y + 1) // rows_per_item)
        scroll = min(
            max(0, self.catalog_type_cursor - visible_items + 1),
            max(0, len(stats) - visible_items),
        )
        visible_stats = stats[scroll : scroll + visible_items]
        for offset, stat in enumerate(visible_stats):
            item_index = scroll + offset
            selected = item_index == self.catalog_type_cursor
            kept = self.filter_editor.catalog_type_kept(
                stat.catalog,
                stat.value,
            )
            state = "YES" if kept else "NO"
            marker = ">" if selected else " "
            fraction = 100.0 * float(stat.fraction)
            main = (
                f"{marker} [{state:<3}] {stat.catalog_label} {stat.value}  "
                f"n={stat.count:,} ({fraction:.2f}%)"
            )
            y = start_y + offset * rows_per_item
            self._add(
                y,
                0,
                main,
                self._styles["selected"] if selected else 0,
            )

            facts: list[str] = []
            if stat.known_variable:
                facts.append("known variable")
            if stat.dipper_contaminant:
                facts.append("dipper contaminant")
            if stat.uncertain:
                facts.append("uncertain catalog label")
            facts.append(stat.description)
            detail = " · ".join(facts)
            detail_lines = textwrap.wrap(
                detail,
                width=max(1, width - 3),
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
            for line_offset, line in enumerate(detail_lines[:2], start=1):
                self._add(y + line_offset, 2, line, self._styles["dim"])

        self._add(
            height - 3,
            0,
            "↑/↓ move · PgUp/PgDn jump · A/Backspace all YES",
        )
        self._add(height - 2, 0, "Enter/Esc returns to Filters; Enter there applies")
        notice_style = self._styles["error"] if self.notice_error else self._styles["dim"]
        self._add(height - 1, 0, self.notice, notice_style)

    def _draw_menu(
        self,
        items: Sequence[MenuItem],
        *,
        selected: set[str],
        start_y: int,
        end_y: int,
    ) -> None:
        _, width = self.screen.getmaxyx()
        available_rows = max(1, end_y - start_y + 1)
        rendered = [f"[{item.key}] {item.label}" for item in items]
        longest = max((len(text) for text in rendered), default=20) + 2
        if width <= PORTRAIT_MAX_WIDTH:
            columns = 2 if width >= 2 * longest else 1
        else:
            columns = max(1, min(3, width // max(1, longest)))
        if columns == 1:
            # A single narrow cell can still be narrower than one exceptionally
            # long taxonomy label. Paginate wrapped entries instead of
            # truncating their final words.
            def pages_for(row_capacity: int) -> list[list[tuple[MenuItem, list[str]]]]:
                pages: list[list[tuple[MenuItem, list[str]]]] = []
                page: list[tuple[MenuItem, list[str]]] = []
                used_rows = 0
                for item, text in zip(items, rendered):
                    lines = textwrap.wrap(
                        text,
                        width=max(1, width - 1),
                        break_long_words=True,
                        break_on_hyphens=False,
                    ) or [""]
                    if page and used_rows + len(lines) > row_capacity:
                        pages.append(page)
                        page = []
                        used_rows = 0
                    page.append((item, lines))
                    used_rows += len(lines)
                if page or not pages:
                    pages.append(page)
                return pages

            pages = pages_for(available_rows)
            if len(pages) > 1 and available_rows > 1:
                rows_for_items = available_rows - 1
                pages = pages_for(rows_for_items)
            self.mode.page = min(self.mode.page, len(pages) - 1)
            y = start_y
            for item, lines in pages[self.mode.page]:
                style = self._styles["selected"] if item.value in selected else 0
                for line in lines:
                    self._add(y, 0, line, style)
                    y += 1
            if len(pages) > 1:
                page_text = f"page {self.mode.page + 1}/{len(pages)}  [PgUp/PgDn]"
                self._add(
                    end_y,
                    max(0, width - len(page_text) - 1),
                    page_text,
                    self._styles["dim"],
                )
            return
        cell_width = max(1, width // columns)
        rows_for_items = available_rows
        page_size = max(1, rows_for_items * columns)
        page_count = max(1, (len(items) + page_size - 1) // page_size)
        if page_count > 1 and available_rows > 1:
            # Reserve the final row for the page indicator instead of drawing it
            # over the last menu item in a short terminal.
            rows_for_items = available_rows - 1
            page_size = max(1, rows_for_items * columns)
            page_count = max(1, (len(items) + page_size - 1) // page_size)
        self.mode.page = min(self.mode.page, page_count - 1)
        page_start = self.mode.page * page_size
        page_items = list(items[page_start:page_start + page_size])
        # Balance a short page across the requested columns instead of filling
        # one very tall left column and leaving the others empty.
        used_columns = min(columns, max(1, len(page_items)))
        item_rows = max(
            1,
            min(
                rows_for_items,
                (len(page_items) + used_columns - 1) // used_columns,
            ),
        )
        for offset, item in enumerate(page_items):
            row = offset % item_rows
            column = offset // item_rows
            x = column * cell_width
            text = f"[{item.key}] {item.label}"
            style = self._styles["selected"] if item.value in selected else 0
            self._add(start_y + row, x, text, style)
        if page_count > 1:
            page_text = f"page {self.mode.page + 1}/{page_count}  [PgUp/PgDn]"
            self._add(end_y, max(0, width - len(page_text) - 1), page_text, self._styles["dim"])

    def _draw_footer(self, start_y: int) -> None:
        """Draw a scannable, keys-first control legend.

        Keys are drawn in the ``key`` colour (cyan) and labels in the neutral
        foreground so the terminal reads like a keymap rather than a wall of
        text.  Nothing is bold — bold characters eat cells that we already
        need for identity and catalog data above.
        """

        _height, width = self.screen.getmaxyx()
        # The main reviewer has one responsive content set at every valid
        # width, so its controls retain the same labels and only reflow.
        portrait = True

        def emit(row: int, groups: Sequence[tuple[str, str]]) -> int:
            col = 0
            for key, label in groups:
                token_width = len(key) + 1 + len(label)
                separator_width = 2 if col else 0
                if col and col + separator_width + token_width > width - 1:
                    row += 1
                    col = 0
                    separator_width = 0
                if separator_width:
                    self._add(row, col, "  ", self._styles["label"])
                    col += 2
                self._add(row, col, key, self._styles["key"])
                col += len(key)
                self._add(row, col, f" {label}", self._styles["value"])
                col += 1 + len(label)
            return row

        last_footer_row = start_y
        if portrait:
            if self.mode.name == "subtypes":
                context = [("Esc", "primary"), ("Backspace", "clear"), ("PgUp/Dn", "pages")]
            elif self.mode.name == "physical":
                context = [("Esc", "primary"), ("Backspace", "clear"), ("1-4", "conf")]
            elif self.mode.name == "physical_subtypes":
                context = [("Esc", "physical"), ("Backspace", "clear"), ("PgUp/Dn", "pages")]
            else:
                context = [("Space", "subtypes"), ("Backspace", "back"), ("1-4", "confidence")]
            emit(start_y, context)
            emit(
                start_y + 1,
                [
                    ("S", "save+next"),
                    (".", "save"),
                    ("N", "next"),
                    (",", "follow-up"),
                ],
            )
            emit(
                start_y + 2,
                [
                    ("F", "filter"),
                    ("H", "hypothesis"),
                    ("M", "notes"),
                    ("V", "VSX"),
                    ("/", "find"),
                ],
            )
            emit(
                start_y + 3,
                [
                    ("-/+", "phase"),
                    ("P", "period"),
                    ("R", "recompute"),
                    ("W", "window"),
                ],
            )
            last_footer_row = emit(
                start_y + 4,
                [
                    ("A", "all cameras"),
                    ("C", "colors"),
                    ("O", "overlays"),
                    ("T", "theme"),
                ],
            )
            last_footer_row = emit(
                start_y + 5,
                [
                    ("E", "events"),
                    ("X", "export"),
                    ("Q", "quit"),
                    ("?", "help"),
                ],
            )
        else:
            if self.mode.name == "subtypes":
                emit(start_y, [("Esc", "primary"), ("Backspace", "clear"), ("PgUp/Dn", "pages")])
            elif self.mode.name == "physical":
                emit(start_y, [("Esc", "primary"), ("Backspace", "clear physical")])
            elif self.mode.name == "physical_subtypes":
                emit(start_y, [("Esc", "physical"), ("Backspace", "clear subtype")])
            else:
                emit(start_y, [("Space", "subtypes"), ("Backspace", "previous")])
            emit(
                start_y + 1,
                [("1-4", "confidence"), ("Enter", "save+next"), (".", "save"), ("Tab", "skip")],
            )
            emit(
                start_y + 2,
                [
                    ("F", "filters"),
                    ("H", "physical"),
                    ("M", "notes"),
                    ("V", "VSX"),
                    ("/", "find"),
                    ("Q", "quit"),
                ],
            )
            last_footer_row = emit(
                start_y + 3,
                [
                    ("-/+", "phase"),
                    ("P", "manual"),
                    ("R", "PDM"),
                    ("W", "window"),
                    ("C", "cams"),
                    ("G", "colors"),
                    ("O", "external"),
                    ("T", "theme"),
                    ("E", "marks"),
                    ("X", "export"),
                    ("?", "help"),
                ],
            )
        notice_row = start_y + (5 if portrait else 4)
        # Suppress the vestigial "Ready" chatter; only surface a notice line
        # when the message is meaningful (an error, a mode change, etc.).
        if (
            notice_row > last_footer_row
            and self.notice
            and self.notice.strip().lower() != "ready"
        ):
            notice_style = (
                self._styles["error"] if self.notice_error else self._styles["dim"]
            )
            self._add(notice_row, 0, self.notice, notice_style)

    def _draw_help(self) -> None:
        height, width = self.screen.getmaxyx()
        source_lines = (
            "MALCA REVIEW HELP",
            "",
            "CLASSIFY",
            "Primary letter: select morphology, then toggle subtypes",
            "Space: reopen subtypes   H: physical hypothesis + subtype",
            "Backspace: previous / clear current subtype selection",
            "1-4: confidence          ,: follow-up   M: notes",
            "V: open VSX coordinate search",
            "",
            "SAVE / MOVE",
            "S: save + next           .: save here",
            "N: next without saving   Q: quit",
            "F: filters               /: find candidate",
            "",
            "IMAGE",
            "-/+: half/double period  P: manual period",
            "R: recompute PDM          W: cycle PDM search window",
            "A: all/cleaned cameras    C: toggle camera colors",
            "O: external overlays      T: plot theme (white/black)",
            "L: full/ASAS-SN time window",
            "E: event markers",
            "X: export PNG next to review DB",
            "Publication-style LC by default; C restores per-camera colors.",
            "Persistent window: LC, phase fold, cutout, Gaia CMD, SED.",
            "",
            "Browser-only internal tags and disposition fields are preserved on save.",
        )
        rendered: list[tuple[str, int]] = []
        for line in source_lines:
            if not line:
                rendered.append(("", 0))
                continue
            style = self._styles["title"] if line.isupper() else 0
            wrapped = textwrap.wrap(
                line,
                width=max(12, width - 1),
                break_long_words=False,
                break_on_hyphens=False,
            ) or [""]
            rendered.extend((part, style) for part in wrapped)
        for row, (line, style) in enumerate(rendered[: max(0, height - 1)]):
            self._add(row, 0, line, style)
        self._add(height - 1, 0, "Press any key to return.", self._styles["dim"])

    def _draw_centered_prompt(self, text: str) -> None:
        height, width = self.screen.getmaxyx()
        box_width = min(max(len(text) + 4, 36), max(8, width - 4))
        y = max(1, height // 2 - 1)
        x = max(1, (width - box_width) // 2)
        content_width = max(1, box_width - 4)
        display_text = str(text)
        if len(display_text) > content_width:
            display_text = "…" + display_text[-(content_width - 1):]
        self._add(y, x, " " * box_width, curses.A_REVERSE)
        self._add(
            y + 1,
            x,
            f"  {display_text}  ".ljust(box_width),
            curses.A_REVERSE | curses.A_BOLD,
        )
        self._add(y + 2, x, " " * box_width, curses.A_REVERSE)

    def _add_clipped(
        self,
        row: int,
        column: int,
        text: object,
        max_width: int,
        style: int = 0,
    ) -> None:
        self._add(row, column, str(text)[: max(0, int(max_width))], style)

    def _add(self, row: int, column: int, text: object, style: int = 0) -> None:
        height, width = self.screen.getmaxyx()
        if row < 0 or row >= height or column < 0 or column >= width:
            return
        value = str(text)
        max_len = max(0, width - column - 1)
        if max_len <= 0:
            return
        try:
            self.screen.addnstr(row, column, value, max_len, style)
        except curses.error:
            pass

    @staticmethod
    def _is_enter(key: object) -> bool:
        return key in {"\n", "\r", curses.KEY_ENTER}

    @staticmethod
    def _is_tab(key: object) -> bool:
        return key in {"\t", curses.KEY_BTAB}

    @staticmethod
    def _is_backspace(key: object) -> bool:
        return key in {"\b", "\x7f", curses.KEY_BACKSPACE}

    @staticmethod
    def _is_escape(key: object) -> bool:
        return key == "\x1b"

    @staticmethod
    def _is_page_down(key: object) -> bool:
        return key == curses.KEY_NPAGE

    @staticmethod
    def _is_page_up(key: object) -> bool:
        return key == curses.KEY_PPAGE


def _cache_dir_for_db(db_path: Path) -> Path:
    digest = hashlib.sha1(str(db_path.expanduser().resolve()).encode("utf-8")).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"malca-review-tui-{digest}"


def _run_curses(
    screen: Any,
    repository: "ReviewRepository",
    images: "ImageCoordinator",
    *,
    db_path: Path,
    run_dir: Path | None,
    external_time_window: str,
    external_time_padding_days: float,
) -> None:
    ReviewTuiApp(
        screen,
        repository,
        images,
        db_path=db_path,
        run_dir=run_dir,
        external_time_window=external_time_window,
        external_time_padding_days=external_time_padding_days,
    ).run()


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    db_path = args.review_db.expanduser()
    if not db_path.exists() or not db_path.is_file():
        raise SystemExit(f"Review database does not exist: {db_path}")
    if args.image_cache_size < 1:
        raise SystemExit("--image-cache-size must be at least 1")
    if (
        not math.isfinite(args.external_time_padding_days)
        or args.external_time_padding_days < 0.0
    ):
        raise SystemExit("--external-time-padding-days must be finite and nonnegative")

    # Keep --help and parser startup lightweight; the store and plotting stacks
    # are imported only after a real TUI run has been requested.
    from malca.review.tui_render import ImageCoordinator
    from malca.review.tui_service import ReviewRepository

    repository = ReviewRepository(
        db_path,
        only_unreviewed=not args.all,
        candidate_query=args.candidate,
        reviewer=args.reviewer,
        external_results_root=args.run_dir,
    )
    images = ImageCoordinator(
        _cache_dir_for_db(db_path),
        viewer=args.viewer,
        cache_size=args.image_cache_size,
    )
    try:
        curses.wrapper(
            _run_curses,
            repository,
            images,
            db_path=db_path,
            run_dir=args.run_dir,
            external_time_window=args.external_time_window,
            external_time_padding_days=args.external_time_padding_days,
        )
    except KeyboardInterrupt:
        return 130
    finally:
        images.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
