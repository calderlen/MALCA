from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from malca.review.tui import ReviewTuiApp, _compact_token_line, build_parser
from malca.review.tui_controller import ReviewDraft
from malca.review.tui_service import (
    CatalogTypeStat,
    NumericRange,
    QueueFilterSpec,
)


class _Images:
    def __init__(self) -> None:
        self.requests = []
        self.prefetches = []
        self.poll_result = None

    def request_current(self, request):
        self.requests.append(request)
        return SimpleNamespace(generation=len(self.requests), state="rendering")

    def prefetch(self, request) -> None:
        self.prefetches.append(request)

    def poll(self):
        return self.poll_result


class _Repository:
    def __init__(self, *, fail_save: bool = False) -> None:
        self.candidate_ids = ("c1", "c2")
        self.start_index = 0
        self.fail_save = fail_save
        self.saved = []
        self.persisted = []
        self.filter_spec = QueueFilterSpec.default()
        self.applied_filters = []
        self.search_override = False

    def load(self, candidate_id: str):
        record = SimpleNamespace(
            candidate_id=candidate_id,
            asas_sn_id=candidate_id.removeprefix("c"),
            lc_path=Path(f"/{candidate_id}.dat3"),
            source_path="/run/results/candidates.parquet",
            payload={"period_consensus_days": 2.0, "ra": 101.25, "dec": -22.5},
        )
        review = {
            "workflow_status": "unreviewed",
            "morphology_primary": None,
            "morphology_secondary_json": "[]",
            "classification_confidence": None,
        }
        return record, review

    def persist_last_candidate(self, candidate_id: str) -> None:
        self.persisted.append(candidate_id)

    def save(self, candidate_id, draft, *, increment_pass, event_type):
        if self.fail_save:
            raise RuntimeError("database busy")
        self.saved.append((candidate_id, increment_pass, event_type, draft.snapshot()))
        return {
            "workflow_status": "reviewed",
            "morphology_primary": draft.morphology_primary,
            "morphology_secondary_json": draft.morphology_secondary_json,
            "classification_confidence": draft.confidence,
            "physical_primary": draft.physical_primary,
        }

    def preview_filter_count(self, spec) -> int:
        return 1 if spec.prob_dipper_dimming.minimum == 0.5 else 2

    def catalog_type_stats(self):
        return (
            CatalogTypeStat(
                catalog="vsx",
                catalog_label="VSX",
                column="vsx_class",
                value="EA",
                count=120,
                total_candidates=200,
                description="Algol-type eclipsing binary",
                known_variable=True,
                dipper_contaminant=True,
                uncertain=False,
            ),
            CatalogTypeStat(
                catalog="gaia",
                catalog_label="Gaia",
                column="gaia_var_class",
                value="YSO",
                count=18,
                total_candidates=200,
                description="Young stellar object",
                known_variable=True,
                dipper_contaminant=True,
                uncertain=False,
            ),
            CatalogTypeStat(
                catalog="asassn",
                catalog_label="ASAS-SN",
                column="asassn_var_type",
                value="VAR",
                count=6,
                total_candidates=200,
                description="Generic variable",
                known_variable=False,
                dipper_contaminant=False,
                uncertain=False,
            ),
            CatalogTypeStat(
                catalog="simbad",
                catalog_label="SIMBAD",
                column="simbad_otype",
                value="Y*O",
                count=42,
                total_candidates=200,
                description="Young stellar object",
                known_variable=False,
                dipper_contaminant=False,
                uncertain=False,
            ),
        )

    def external_photometry_counts(self):
        return {
            "neowise": 40,
            "ztf": 25,
            "gaia_epoch": 12,
        }

    def apply_filters(self, spec, *, anchor_candidate_id=None) -> int:
        self.filter_spec = spec
        self.applied_filters.append((spec, anchor_candidate_id))
        self.candidate_ids = ("c2",)
        self.search_override = False
        return 0

    def search_candidate(self, query: str):
        if query not in {"c1", "c2", "1", "2"}:
            raise ValueError("No candidate matches")
        candidate_id = query if query.startswith("c") else f"c{query}"
        if candidate_id in self.candidate_ids:
            return self.candidate_ids.index(candidate_id), False
        self.candidate_ids = (candidate_id,)
        self.search_override = True
        return 0, True


class _Screen:
    """Small curses-compatible screen used for semantic layout assertions."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.refresh_count = 0
        self.erase()

    def getmaxyx(self):
        return self.height, self.width

    def erase(self) -> None:
        self.cells = [[" "] * self.width for _ in range(self.height)]

    def addnstr(self, row, column, value, max_len, _style=0) -> None:
        assert 0 <= row < self.height
        assert 0 <= column < self.width
        for offset, character in enumerate(str(value)[:max_len]):
            target = column + offset
            if target >= self.width:
                break
            self.cells[row][target] = character

    def refresh(self) -> None:
        self.refresh_count += 1

    def text(self) -> str:
        return "\n".join("".join(row).rstrip() for row in self.cells)

    def row(self, index: int) -> str:
        return "".join(self.cells[index]).rstrip()


def _app(*, fail_save: bool = False) -> ReviewTuiApp:
    app = ReviewTuiApp(
        None,
        _Repository(fail_save=fail_save),
        _Images(),
        db_path=Path("review.db"),
        run_dir=Path("/run"),
    )
    app.image_status = "ready"
    return app


def _drawn_app(width: int, height: int) -> tuple[ReviewTuiApp, _Screen]:
    app = _app()
    screen = _Screen(width, height)
    app.screen = screen
    app._styles = {
        "title": 0,
        "section": 0,
        "label": 0,
        "value": 0,
        "key": 0,
        "catalog": 0,
        "selected": 0,
        "dirty": 0,
        "dim": 0,
        "error": 0,
        "link": 0,
    }
    app.record = SimpleNamespace(
        candidate_id="stv_68719905680",
        asas_sn_id="68719905680",
        lc_path=Path("/68719905680.dat3"),
        source_path="/run/results/candidates.parquet",
        payload={
            "phot_g_mean_mag": 9.87,
            "bp_rp": 3.49,
            "ruwe": 1.28,
            "prob_hierarchical_artifact_or_nonvariable": 0.08,
            "prob_usable_astrophysical_variable": 0.92,
            "prob_brightening_transient": 0.05,
            "prob_dipper_dimming": 0.25,
            "prob_eb_geometric_periodic": 0.15,
            "prob_long_period_variable_hierarchical": 0.05,
            "prob_long_term_variable_hierarchical": 0.05,
            "prob_long_timescale_variable": 0.10,
            "prob_microlensing_hierarchical": 0.20,
            "prob_other_structured_variable": 0.37,
            "prob_quasi_periodic_hierarchical": 0.25,
            "prob_recurrent_dipper_hierarchical": 0.12,
            "prob_single_dipper_hierarchical": 0.13,
            "dipper_score": 1.1,
            "jumper_score": 23.1,
            "stats_variability_quasi_periodicity_q": 0.68,
            "stats_variability_flux_asymmetry_m": 0.02,
            "vsx_class": "L",
            "gaia_var_class": "LPV",
            "vetting_likely_known": True,
            "pm_total": 132.4,
            "ra": 101.25,
            "dec": -22.5,
            "source_id": "1234567890123456789",
        },
    )
    app.vsx_url = "https://vsx.aavso.org/index.php?test=1"
    app.review = {"workflow_status": "unreviewed"}
    app.draft = ReviewDraft.from_review(app.review)
    app.phase_period_days = 8.103976
    app.phase_source = "Auto PDM"
    app.survey_label = "DSS2 (DECaPS DR2 fallback)"
    app.notice = "Ready"
    app._draw()
    return app, screen


def test_parser_defaults_to_unreviewed_persistent_window_session() -> None:
    args = build_parser().parse_args(["--review-db", "review.db"])

    assert args.review_db == Path("review.db")
    assert args.all is False
    assert args.viewer == "window"
    assert args.reviewer == "calder"
    assert args.external_time_window == "full"
    assert args.external_time_padding_days == 0.0


def test_parser_configures_asassn_time_window() -> None:
    args = build_parser().parse_args(
        [
            "--review-db",
            "review.db",
            "--external-time-window",
            "asassn",
            "--external-time-padding-days",
            "30",
        ]
    )

    assert args.external_time_window == "asassn"
    assert args.external_time_padding_days == 30.0


def test_render_request_uses_only_browser_authoritative_period_fields() -> None:
    app = _app()
    record = SimpleNamespace(
        candidate_id="c1",
        asas_sn_id="1",
        lc_path=Path("/c1.dat3"),
        source_path="/run/results/candidates.parquet",
        payload={
            "period_consensus_days": 2.5,
            "periodicity_period": 9.0,
        },
    )

    request = app._render_request(record)

    assert request.payload == record.payload
    assert request.payload is not record.payload
    assert request.stored_phase_period_days == 2.5
    assert request.stored_phase_source == "period_consensus_days"

    record.payload = {"periodicity_period": 9.0}
    request = app._render_request(record)
    assert request.stored_phase_period_days is None
    assert request.stored_phase_source is None


def test_primary_key_enters_subtype_mode_and_subtypes_are_multi_select() -> None:
    app = _app()

    app._handle_primary_key("e")
    assert app.mode.name == "subtypes"
    assert app.draft.morphology_primary == "dimming_event"

    app._handle_subtype_key("k")
    app._handle_subtype_key("h")
    assert app.draft.morphology_secondaries == [
        "recurrent_dips",
        "asymmetric_dip",
    ]

    app._handle_subtype_key("k")
    assert app.draft.morphology_secondaries == ["asymmetric_dip"]


def test_backspace_clears_subtypes_in_subtype_mode() -> None:
    app = _app()
    app._handle_primary_key("e")
    app._handle_subtype_key("k")

    app._handle_key("\x7f")

    assert app.index == 0
    assert app.draft.morphology_secondaries == []
    assert app.mode.name == "subtypes"


def test_save_failure_keeps_candidate_and_dirty_draft() -> None:
    app = _app(fail_save=True)
    app.draft.select_primary("dimming_event")
    app.draft.set_confidence(3)

    app._save(advance=True)

    assert app.index == 0
    assert app.draft.dirty is True
    assert app.notice_error is True
    assert "database busy" in app.notice


def test_successful_done_saves_before_loading_next_candidate() -> None:
    app = _app()
    app.record, app.review = app.repository.load("c1")
    app.draft = ReviewDraft.from_review(app.review)
    app.draft.select_primary("dimming_event")
    app.draft.toggle_subtype("recurrent_dips")
    app.draft.set_confidence(4)

    app._save(advance=True)

    assert app.repository.saved[0][:3] == ("c1", True, "tui_done")
    assert app.index == 1
    assert app.record.candidate_id == "c2"
    assert app.repository.persisted == []
    assert app.notice == "Saved + next"


def test_label_keys_are_blocked_until_current_image_is_ready() -> None:
    app = _app()
    app.image_status = "rendering"

    app._handle_key("e")

    assert app.draft.morphology_primary is None
    assert app.notice_error is True
    assert "Wait for the current image" in app.notice

    app.image_status = "ready"
    app._poll_images()
    app._handle_key("e")
    assert app.draft.morphology_primary == "dimming_event"


def test_save_debounce_does_not_eat_a_legitimate_fast_next_candidate_save() -> None:
    app = _app()
    app.record, app.review = app.repository.load("c1")
    app.draft = ReviewDraft.from_review(app.review)
    app.draft.select_primary("dimming_event")
    app.draft.set_confidence(3)

    app._save(advance=True)
    app.image_status = "ready"
    app.draft.select_primary("dimming_event")
    app.draft.set_confidence(3)
    app._save(advance=True)

    assert len(app.repository.saved) == 2
    assert app.index == 1
    assert "final candidate" in app.notice.lower()


def test_save_debounce_still_blocks_a_repeat_on_the_same_candidate() -> None:
    app = _app()
    app.record, app.review = app.repository.load("c1")
    app.draft = ReviewDraft.from_review(app.review)
    app.draft.select_primary("dimming_event")
    app.draft.set_confidence(3)

    app._save(advance=False)
    app._save(advance=False)

    assert len(app.repository.saved) == 1
    assert "debounce" in app.notice.lower()


def test_clean_reviewed_candidate_cannot_be_resaved_by_late_key_repeat() -> None:
    app = _app()
    app.review = {
        "workflow_status": "reviewed",
        "morphology_primary": "dimming_event",
        "morphology_secondary_json": "[]",
        "classification_confidence": 3,
    }
    app.draft = ReviewDraft.from_review(app.review)
    app._last_save_completed_at = float("-inf")

    app._save(advance=True)

    assert app.repository.saved == []
    assert app.index == 0
    assert "unchanged" in app.notice.lower()


def test_navigation_reports_when_dirty_draft_is_discarded() -> None:
    app = _app()
    app.draft.select_primary("dimming_event")

    app._navigate(1, "Skipped without saving")

    assert app.index == 1
    assert app.notice == "Skipped without saving; unsaved changes discarded"


def test_resume_position_is_persisted_best_effort_at_session_end() -> None:
    app = _app()
    app.index = 1

    app._persist_position_best_effort()

    assert app.repository.persisted == ["c2"]


def test_physical_hypothesis_opens_its_subtypes() -> None:
    app = _app()

    app._handle_key("H")
    app._handle_key("y")

    assert app.mode.name == "physical_subtypes"
    assert app.draft.physical_primary == "young_stellar_object_or_pms"
    assert app.draft.dirty is True


def test_physical_subtype_and_notes_are_editable() -> None:
    app = _app()

    app._handle_key("H")
    app._handle_key("p")
    assert app.mode.name == "physical_subtypes"
    app._handle_key("s")
    assert app.draft.physical_secondary == "rr_lyrae"

    app._handle_key("M")
    app.input_buffer = "reviewed in terminal"
    app._handle_key("\n")
    assert app.draft.notes == "reviewed in terminal"
    assert app.draft.dirty is True


def test_vsx_keyboard_action_opens_canonical_candidate_link(monkeypatch) -> None:
    app = _app()
    app._load_current()
    opened = []
    monkeypatch.setattr(
        "malca.review.tui.webbrowser.open_new_tab",
        lambda url: opened.append(url) or True,
    )

    app._handle_key("V")

    assert opened == [app.vsx_url]
    assert "vsx.aavso.org" in opened[0]


def test_filter_apply_requires_explicit_dirty_draft_decision() -> None:
    app = _app()
    app.draft.select_primary("dimming_event")
    app._handle_key("F")
    app.filter_editor.spec = QueueFilterSpec(
        prob_dipper_dimming=NumericRange(0.5),
        sort_by="prob_dipper_dimming",
        sort_desc=True,
    )

    app._handle_key("\n")
    assert app.mode.name == "filter_confirm"
    assert app.repository.applied_filters == []

    app._handle_key("d")
    assert len(app.repository.applied_filters) == 1
    assert app.candidate_ids == ("c2",)
    assert app.record.candidate_id == "c2"
    assert "Filters applied" in app.notice


def test_filter_configures_external_photometry_master_and_sources() -> None:
    app = _app()
    app._load_current()
    app._handle_key("F")

    while app.filter_editor.active_key != "show_external_lightcurves":
        app.filter_editor.move(1)
    app._handle_key(" ")
    assert app.filter_editor.spec.show_external_lightcurves is False

    while app.filter_editor.active_key != "external_source_ztf":
        app.filter_editor.move(1)
    app._handle_key(" ")
    assert "ztf" not in app.filter_editor.spec.external_lightcurve_sources

    while app.filter_editor.active_key != "external_source_gaia_epoch":
        app.filter_editor.move(1)
    app._handle_key(" ")
    assert "gaia_epoch" in app.filter_editor.spec.external_lightcurve_sources

    while app.filter_editor.active_key != "external_availability_neowise":
        app.filter_editor.move(1)
    app._handle_key(" ")
    assert app.filter_editor.spec.required_external_photometry_sources == (
        "neowise",
    )

    app._handle_key("\n")

    assert app.show_external_lightcurves is False
    assert "ztf" not in app.external_lightcurve_sources
    assert "gaia_epoch" in app.external_lightcurve_sources
    assert app.repository.filter_spec.show_external_lightcurves is False
    assert app.repository.filter_spec.required_external_photometry_sources == (
        "neowise",
    )
    assert app.images.requests[-1].show_external_lightcurves is False
    assert app.images.requests[-1].external_lightcurve_sources == (
        app.external_lightcurve_sources
    )


def test_catalog_type_menu_toggles_keep_state_and_applies_filter() -> None:
    app = _app()
    app._handle_key("F")
    while app.filter_editor.active_key != "catalog_vsx":
        app.filter_editor.move(1)

    app._handle_key(" ")
    assert app.mode.name == "filter_catalog_types"
    assert len(app.catalog_type_stats) == 4
    assert len(app._active_catalog_type_stats()) == 1

    app._handle_key("n")
    assert app.filter_editor.spec.excluded_vsx_types == ("EA",)
    assert "NO / exclude" in app.notice

    app._handle_key("\n")
    assert app.mode.name == "filters"
    app._handle_key("\n")

    assert app.repository.applied_filters
    assert app.repository.applied_filters[-1][0].excluded_vsx_types == ("EA",)


def test_catalog_type_menu_renders_counts_descriptions_and_risk_facts() -> None:
    app, screen = _drawn_app(80, 40)
    app._handle_key("F")
    while app.filter_editor.active_key != "catalog_vsx":
        app.filter_editor.move(1)
    app._handle_key(" ")

    app._draw()
    rendered = screen.text()

    assert "VSX TYPES" in rendered
    assert "[YES] VSX EA" in rendered
    assert "Gaia YSO" not in rendered
    assert "n=120 (60.00%)" in rendered
    assert "known variable" in rendered
    assert "dipper contaminant" in rendered
    assert "Algol-type eclipsing binary" in rendered


def test_each_populated_catalog_opens_its_own_type_menu() -> None:
    app, screen = _drawn_app(80, 40)
    app._handle_key("F")
    assert {
        row.key for row in app.filter_editor.rows() if row.kind == "catalog"
    } == {
        "catalog_vsx",
        "catalog_gaia",
        "catalog_asassn",
        "catalog_simbad",
    }

    while app.filter_editor.active_key != "catalog_simbad":
        app.filter_editor.move(1)
    app._handle_key(" ")
    app._draw()
    rendered = screen.text()

    assert "SIMBAD TYPES" in rendered
    assert "[YES] SIMBAD Y*O" in rendered
    assert "VSX EA" not in rendered

    app._handle_key("n")
    assert app.filter_editor.spec.excluded_simbad_types == ("Y*O",)
    app._handle_key("A")
    assert app.filter_editor.spec.excluded_simbad_types == ()


def test_empty_queue_can_recover_by_applying_filters() -> None:
    app = _app()
    app.repository.candidate_ids = ()
    app.index = 0

    app._handle_key("F")
    app._handle_key("\n")

    assert app.candidate_ids == ("c2",)
    assert app.record.candidate_id == "c2"


def test_search_requires_dirty_confirmation_and_can_open_global_candidate() -> None:
    app = _app()
    app.repository.candidate_ids = ("c1",)
    app.draft.select_primary("dimming_event")

    app._handle_key("/")
    app._handle_key("2")
    app._handle_key("\n")
    assert app.mode.name == "search_confirm"
    assert app.candidate_ids == ("c1",)

    app._handle_key("d")
    assert app.record.candidate_id == "c2"
    assert app.repository.search_override is True
    assert "Jumped" in app.notice


def test_phase_and_image_controls_are_forwarded_to_renderer() -> None:
    app = _app()
    app._load_current()
    app.image_status = "ready"

    app._handle_key("+")
    assert app.images.requests[-1].phase_multiplier == 2.0

    app._handle_key("A")
    assert app.images.requests[-1].camera_view == "all"

    app._handle_key("E")
    assert app.images.requests[-1].show_event_markers is True
    assert app.images.prefetches[-1].camera_view == "all"
    assert app.images.prefetches[-1].show_event_markers is True

    app._handle_key("O")
    assert app.images.requests[-1].show_external_lightcurves is False
    assert app.images.prefetches[-1].show_external_lightcurves is False
    assert app.images.requests[-1].external_lightcurve_sources == (
        "atlas",
        "ztf",
        "neowise",
        "asas3",
        "crts",
        "dasch",
    )

    app.asassn_window_padding_days = 30.0
    app._handle_key("L")
    assert app.images.requests[-1].time_window_mode == "asassn"
    assert app.images.requests[-1].asassn_window_padding_days == 30.0
    assert app.images.prefetches[-1].time_window_mode == "asassn"
    assert "ASAS-SN span" in app.notice

    app._handle_key("P")
    app.input_buffer = "0.312131"
    app._handle_key("\n")
    assert app.images.requests[-1].manual_phase_period_days == 0.312131
    assert app.images.requests[-1].manual_phase_source == "Manual"

    app._handle_key("R")
    assert app.images.requests[-1].force_period_search is True
    assert app.images.requests[-1].manual_phase_period_days is None

    first_token = app.images.requests[-1].force_period_search_token
    app._handle_key("R")
    assert app.images.requests[-1].force_period_search_token != first_token


def test_plot_theme_defaults_to_dark_and_can_toggle_to_light() -> None:
    app = _app()
    app._load_current()

    assert app.plot_theme == "black"
    app._handle_key("T")

    assert app.plot_theme == "white"
    assert app.images.requests[-1].plot_theme == "white"


def test_resolved_automatic_period_is_reused_until_camera_policy_changes() -> None:
    app = _app()
    app._load_current()
    generation = app._current_image_generation
    app.images.poll_result = SimpleNamespace(
        generation=generation,
        state="ready",
        error=None,
        phase_period_days=0.31213094,
        phase_source="Auto PDM",
        survey_label="DSS2 (DECaPS DR2 fallback)",
    )

    app._poll_images()
    assert app.manual_phase_period_days == pytest.approx(0.31213094)
    assert app.manual_phase_source == "Auto PDM"

    app._handle_key("+")
    assert app.images.requests[-1].manual_phase_period_days == pytest.approx(0.31213094)
    assert app.images.requests[-1].phase_multiplier == 2.0

    app._handle_key("C")
    assert app.images.requests[-1].color_by_camera is True

    app._handle_key("A")
    assert app.images.requests[-1].manual_phase_period_days is None


def test_ready_image_without_period_clears_forced_search_running_state() -> None:
    app = _app()
    app._load_current()
    app._handle_key("R")
    generation = app._current_image_generation
    app.images.poll_result = SimpleNamespace(
        generation=generation,
        state="ready",
        error=None,
        phase_period_days=None,
        phase_source=None,
        survey_label="DECaPS DR2",
    )

    app._poll_images()

    assert app.force_period_search is False
    assert app.force_period_search_token is None
    assert app.phase_period_days is None
    assert app.phase_source == "Auto period unavailable"


@pytest.mark.parametrize("width", [48, 60, 80])
@pytest.mark.parametrize("height", [32, 40, 50])
def test_portrait_layout_keeps_identity_details_and_controls_visible(
    width: int, height: int
) -> None:
    _app_instance, screen = _drawn_app(width, height)
    rendered = screen.text()

    assert "Terminal too small" not in rendered
    assert "ASAS-SN: 68719905680" in rendered
    assert "GAIA: 1234567890123456789" in rendered
    assert "P 8.1 d" in rendered
    assert "PDM" in rendered
    assert "cam cleaned" in rendered or "cleaned" in rendered
    assert "colors off" in rendered or "col−" in rendered
    assert "events off" in rendered or "ev−" in rendered
    assert "DRAFT" in rendered
    assert "SCORES" in rendered
    assert "STAR" in rendered or "FLAGS" in rendered
    assert "P " in rendered and ".25" in rendered
    assert "mean mag" in rendered
    assert "RUWE" in rendered and "1.28" in rendered
    assert "Known yes" in rendered
    assert "PM 132.4" in rendered or "PM 132" in rendered
    assert "dip 1.1" not in rendered
    assert "jump 23.1" not in rendered
    assert "RA 101.250000" not in rendered
    assert "Dec -22.500000" not in rendered
    assert "DSS2 fallback" not in rendered
    assert "S save+next" in rendered
    assert "N next" in rendered
    assert "F filter" in rendered
    assert "? help" in rendered
    assert "Q quit" in rendered
    assert "P period" in rendered
    assert "A all cameras" in rendered
    assert "C colors" in rendered
    assert "E events" in rendered


def test_portrait_catalog_entries_wrap_without_dropping_simbad() -> None:
    app, screen = _drawn_app(48, 50)
    app.record.payload["simbad_main_id"] = (
        "candidate uncertain long-period variable classification"
    )

    app._draw()
    rendered = screen.text()

    assert "VSX L" in rendered
    assert "Gaia LPV" in rendered
    assert "SIMBAD candidate uncertain long-period" in rendered
    assert "variable" in rendered
    assert "classification" in rendered


def test_portrait_catalog_line_shows_asassn_variable_type_and_period() -> None:
    app, screen = _drawn_app(48, 50)
    app.record.payload.update(
        {
            "asassn_var_type": "YSO",
            "asassn_var_period": 12.34567,
        }
    )

    app._draw()

    assert "ASAS-SN YSO P=12.3457d" in screen.text()


@pytest.mark.parametrize("width", [48, 80, 120])
def test_portrait_width_only_reflows_review_content(width: int) -> None:
    app, screen = _drawn_app(width, 50)
    app.record.payload["simbad_main_id"] = (
        "candidate uncertain long-period variable classification"
    )

    app._draw()
    rendered = screen.text()

    for text in (
        "SIMBAD candidate uncertain long-period",
        "classification",
        "colors off",
        "external ATLAS",
        "events off",
        "theme",
        "black",
        "Q quit",
        "? help",
    ):
        assert text in rendered


@pytest.mark.parametrize(
    ("width", "height"),
    [(47, 50), (80, 25), (44, 40)],
)
def test_unsupported_terminal_sizes_show_complete_minimum(
    width: int, height: int
) -> None:
    _app_instance, screen = _drawn_app(width, height)

    assert "Terminal too small" in screen.text()
    assert "Minimum: 48x32" in screen.text()


def test_too_small_screen_keeps_dirty_quit_confirmation_visible() -> None:
    app, screen = _drawn_app(44, 40)
    app.draft.select_primary("dimming_event")
    app.mode.name = "quit"

    app._draw()

    assert "Dirty draft: y quit  n return" in screen.text()


def test_filter_summary_compaction_preserves_queue_and_sort_tokens() -> None:
    summary = _compact_token_line(
        (
            "unreviewed",
            "dip",
            "known-",
            "known-neighbor-",
            "contaminant-",
            "Pdip>=.25",
            "Pdip↓",
        ),
        47,
    )

    assert summary.startswith("unreviewed")
    assert "+" in summary
    assert summary.endswith("Pdip↓")
    assert len(summary) <= 47


def test_tui_displays_all_hierarchical_model_scores() -> None:
    _app_instance, screen = _drawn_app(80, 50)
    rendered = screen.text()

    assert "ML CLASS SCORES" in rendered
    for score in (
        "Reject .08",
        "Usable .92",
        "Dipper .25",
        "EB .15",
        "Long .10",
        "Bright .05",
        "Other .37",
        "QP .25",
        "Micro .20",
        "LPV .05",
        "LTV .05",
        "Recur .12",
        "Single .13",
    ):
        assert score in rendered


def test_portrait_menu_uses_one_column_at_60_and_two_at_80() -> None:
    _app_60, screen_60 = _drawn_app(60, 50)
    primary_rows_60 = [
        index
        for index, row in enumerate(screen_60.text().splitlines())
        if "[q] artifact" in row or "[w] nonvariable" in row
    ]
    assert len(set(primary_rows_60)) == 2

    _app_80, screen_80 = _drawn_app(80, 50)
    primary_rows_80 = [
        index
        for index, row in enumerate(screen_80.text().splitlines())
        if "[q] artifact" in row or "[u] quasi-periodic" in row
    ]
    assert primary_rows_80
    assert "[q] artifact" in screen_80.row(primary_rows_80[0])
    assert "[y] periodic" in screen_80.row(primary_rows_80[0])


def test_portrait_filter_overlay_fits_minimum_terminal() -> None:
    app, screen = _drawn_app(48, 32)
    app.mode.name = "filters"
    app.filter_editor.spec = QueueFilterSpec(
        queue_state="unreviewed",
        known_objects="exclude",
        exclude_dipper_contaminants=True,
        sort_by="prob_dipper_dimming",
        sort_desc=True,
    )
    app._draw()
    rendered = screen.text()

    assert "FILTERS" in rendered
    assert "matches" in rendered
    assert "Pdip↓" in rendered
    assert "Enter apply" in rendered
    assert "R reset" in rendered
    assert "Esc cancel" in rendered

    while app.filter_editor.active_key != "sort_desc":
        app.filter_editor.move(1)
    app._draw()
    assert "> Direction:" in screen.text()
    assert "descending" in screen.text()


def test_filter_overlay_renders_hierarchy_sections_and_conditional_controls() -> None:
    app, screen = _drawn_app(80, 32)
    app.mode.name = "filters"

    while (
        app.filter_editor.active_key
        != "prob_microlensing_given_brightening"
    ):
        app.filter_editor.move(1)
    app._draw()
    rendered = screen.text()
    assert "ML CONDITIONAL HEADS" in rendered
    assert "> ML P(micro | bright):" in rendered

    while app.filter_editor.active_key != "predicted_hierarchical_class":
        app.filter_editor.move(1)
    app._draw()
    rendered = screen.text()
    assert "ML PREDICTED CLASSES" in rendered
    assert "> ML hierarchy class:" in rendered


def test_filter_overlay_renders_external_photometry_source_controls() -> None:
    app, screen = _drawn_app(80, 32)
    app._handle_key("F")

    while app.filter_editor.active_key != "external_source_gaia_epoch":
        app.filter_editor.move(1)
    app._draw()
    rendered = screen.text()

    assert "PHOTOMETRY DISPLAY (ASAS-SN ALWAYS ON)" in rendered
    assert "External photometry:" in rendered
    assert "> Gaia epoch:" in rendered
    assert "disabled" in rendered

    while app.filter_editor.active_key != "external_availability_neowise":
        app.filter_editor.move(1)
    app._draw()
    rendered = screen.text()
    assert "PHOTOMETRY AVAILABILITY FILTERS (AND WITH ML)" in rendered
    assert "> NEOWISE W1/W2:" in rendered
    assert "any · n=40" in rendered
