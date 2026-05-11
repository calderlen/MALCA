"""Main window for the PyQt review GUI (test module)."""
from __future__ import annotations

import os
from pathlib import Path

# Qt: prefer PySide6, fall back to PyQt6
try:
    from PySide6.QtCore import Qt, QUrl, QThread, Signal, QTimer
    from PySide6.QtGui import QShortcut, QKeySequence, QPixmap, QImage, QAction
    from PySide6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QTextEdit, QScrollArea, QFrame,
        QMessageBox, QStatusBar, QSplitter, QTableWidget, QTableWidgetItem,
        QCheckBox, QComboBox, QGroupBox, QPlainTextEdit,         QFileDialog, QInputDialog, QDialog, QDialogButtonBox, QFormLayout,
        QMenuBar, QMenu, QApplication, QTreeWidget, QTreeWidgetItem,
        QHeaderView, QAbstractItemView, QSlider, QTabWidget,
    )
    try:
        from PySide6.QtWebEngineWidgets import QWebEngineView
        _HAS_WEBENGINE = True
    except ImportError:
        QWebEngineView = None
        _HAS_WEBENGINE = False
except ImportError:
    from PyQt6.QtCore import Qt, QUrl, QThread, QTimer
    from PyQt6.QtCore import pyqtSignal as Signal
    from PyQt6.QtGui import QShortcut, QKeySequence, QPixmap, QImage, QAction
    from PyQt6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QTextEdit, QScrollArea, QFrame,
        QMessageBox, QStatusBar, QSplitter, QTableWidget, QTableWidgetItem,
        QCheckBox, QComboBox, QGroupBox, QPlainTextEdit,         QFileDialog, QInputDialog, QDialog, QDialogButtonBox, QFormLayout,
        QMenuBar, QMenu, QApplication, QTreeWidget, QTreeWidgetItem,
        QHeaderView, QAbstractItemView, QSlider, QTabWidget,
    )
    try:
        from PyQt6.QtWebEngineWidgets import QWebEngineView
        _HAS_WEBENGINE = True
    except ImportError:
        QWebEngineView = None
        _HAS_WEBENGINE = False

import json
import pandas as pd

from malca.review.store import (
    db_connect,
    query_queue,
    get_candidate_payload,
    get_review,
    save_review,
    replace_candidate_payload_fields,
    get_diagnostic_background,
    import_candidates,
    export_reviews,
    merge_vetting_results,
    load_app_state,
    save_app_state,
)
from malca.review_qt.filters import FilterSidebarWidget
from malca.review.keyboard import (
    handle_key_action,
    KEYBOARD_SHORTCUTS,
    CLASS_KEY_MAP,
    HELP_TEXT,
)
from malca.review.interactive_plot import build_interactive_lightcurve_figure
from malca.review import diagnostic_plots


# Default DB path when --review-db not set
def _default_db_path() -> Path:
    from malca.review.store import DEFAULT_DB_PATH
    return Path(os.environ.get("MALCA_REVIEW_DB", str(DEFAULT_DB_PATH)))


# Background worker for pipeline run
class PipelineWorker(QThread):
    finished = Signal()
    progress = Signal(str)

    def __init__(self, db_path: Path, candidate_id: str, *, force_rerun: bool = False, run_all_missing: bool = False, candidate_ids: list[str] | None = None) -> None:
        super().__init__()
        self._db_path = Path(db_path)
        self._candidate_id = candidate_id
        self._force_rerun = force_rerun
        self._run_all_missing = run_all_missing
        self._candidate_ids = candidate_ids or []
        self.error: str | None = None

    def run(self) -> None:
        try:
            conn = db_connect(self._db_path)
            from malca.review.pipeline import run_missing_stages, STAGE_SIGNATURES
            def progress(msg: str) -> None:
                self.progress.emit(msg)
            if self._run_all_missing and self._candidate_ids:
                for i, cid in enumerate(self._candidate_ids):
                    self.progress.emit(f"[{i+1}/{len(self._candidate_ids)}] {cid}")
                    run_missing_stages(conn, cid, progress_callback=progress)
            else:
                force = list(STAGE_SIGNATURES.keys()) if self._force_rerun else None
                only_force = bool(self._force_rerun)
                run_missing_stages(conn, self._candidate_id, progress_callback=progress, force_stages=force, only_force=only_force)
        except Exception as e:
            self.error = str(e)
        else:
            self.error = None
        self.finished.emit()


# Diagnostic plot builders (name for tab, builder function)
DIAGNOSTIC_BUILDERS = [
    ("CMD", getattr(diagnostic_plots, "build_cmd_figure", None)),
    ("IR color-color", getattr(diagnostic_plots, "build_ir_colorcolor_figure", None)),
    ("Kiel", getattr(diagnostic_plots, "build_kiel_figure", None)),
    ("RPM", getattr(diagnostic_plots, "build_rpm_figure", None)),
    ("UV-Optical", getattr(diagnostic_plots, "build_uv_optical_figure", None)),
    ("Periodicity", getattr(diagnostic_plots, "build_periodicity_plane_figure", None)),
    ("Score balance", getattr(diagnostic_plots, "build_score_balance_figure", None)),
    ("Catalog support", getattr(diagnostic_plots, "build_catalog_support_figure", None)),
    ("Recurrence", getattr(diagnostic_plots, "build_recurrence_regularity_figure", None)),
    ("Dip repeat", getattr(diagnostic_plots, "build_dip_repeatability_figure", None)),
    ("Variability", getattr(diagnostic_plots, "build_variability_strength_figure", None)),
    ("Stetson", getattr(diagnostic_plots, "build_stetson_scatter_figure", None)),
    ("Shape impulsiveness", getattr(diagnostic_plots, "build_shape_impulsiveness_figure", None)),
    ("Harmonic quality", getattr(diagnostic_plots, "build_harmonic_quality_figure", None)),
    ("Autocorr memory", getattr(diagnostic_plots, "build_autocorr_memory_figure", None)),
    ("Cluster astrometry", getattr(diagnostic_plots, "build_cluster_astrometry_figure", None)),
    ("Classifier plane", getattr(diagnostic_plots, "build_classifier_plane_figure", None)),
    ("ATLAS range", getattr(diagnostic_plots, "build_atlas_range_figure", None)),
    ("ZTF range", getattr(diagnostic_plots, "build_ztf_range_figure", None)),
    ("NEOWISE range", getattr(diagnostic_plots, "build_neowise_range_figure", None)),
    ("Gaia epoch", getattr(diagnostic_plots, "build_gaia_epoch_figure", None)),
    ("LTV trend", getattr(diagnostic_plots, "build_ltv_trend_figure", None)),
    ("NEOWISE trend", getattr(diagnostic_plots, "build_neowise_trend_figure", None)),
]


class ReviewMainWindow(QMainWindow):
    """Single-window review UI: queue, one candidate view, plot, save."""

    def __init__(
        self,
        db_path: Path | None = None,
        plot_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self._db_path = Path(db_path) if db_path else _default_db_path()
        self._plot_dir = Path(plot_dir) if plot_dir else None
        self._conn = None
        self._candidate_ids: list[str] = []
        self._queue_df: pd.DataFrame = pd.DataFrame()
        self._current_idx = 0
        self._current_score: int | None = None
        self._current_class = "unclassified"
        self._review_pass = 1
        self._theme = "black"
        self._needs_followup = False
        self._diagnostic_background: dict | None = None
        self._last_lc_fig = None
        self._sidebar_collapsed = False
        self._filter_params: dict = {}
        self._setup_ui()
        self._connect_db()
        self._load_filter_persistence()
        self._load_queue()
        self._install_shortcuts()
        self._apply_theme_stylesheet()
        self._refresh_current_candidate()
        self._metrics_timer = QTimer(self)
        self._metrics_timer.timeout.connect(self._update_metrics)
        self._metrics_timer.start(1000)

    def _setup_ui(self) -> None:
        self.setWindowTitle("MALCA Review (Qt)")
        self.setMinimumSize(900, 600)
        self.resize(1200, 800)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)

        # ---------- Menu bar (Phase 4) ----------
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        import_act = QAction("Import candidates...", self)
        import_act.triggered.connect(self._on_import_candidates)
        file_menu.addAction(import_act)
        export_act = QAction("Export reviews...", self)
        export_act.triggered.connect(self._on_export_reviews)
        file_menu.addAction(export_act)
        merge_act = QAction("Merge vetting results...", self)
        merge_act.triggered.connect(self._on_merge_vetting)
        file_menu.addAction(merge_act)
        file_menu.addSeparator()
        download_run_config_act = QAction("Download run config (current)...", self)
        download_run_config_act.triggered.connect(self._on_download_run_config)
        file_menu.addAction(download_run_config_act)
        tools_menu = menubar.addMenu("Tools")
        cone_act = QAction("Cone search...", self)
        cone_act.triggered.connect(self._on_cone_search)
        tools_menu.addAction(cone_act)

        # Header: progress, candidate id, theme, pipeline
        header = QHBoxLayout()
        header.setSpacing(12)
        self._progress_label = QLabel("[0/0]")
        self._progress_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        self._candidate_label = QLabel("—")
        self._candidate_label.setObjectName("candidateLabel")
        self._candidate_label.setStyleSheet("font-size: 13px; font-weight: 500;")
        header.addWidget(self._progress_label)
        header.addWidget(QLabel("  "))
        header.addWidget(self._candidate_label)
        header.addStretch()
        # Theme combo
        header.addWidget(QLabel("Theme:"))
        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["Black", "White"])
        self._theme_combo.setCurrentText("Black")
        self._theme_combo.currentTextChanged.connect(self._on_theme_changed)
        header.addWidget(self._theme_combo)
        # Pipeline: Run current / Re-run / Run All Missing
        self._run_pipeline_btn = QPushButton("Run pipeline (current)")
        self._run_pipeline_btn.setObjectName("runPipelineBtn")
        self._run_pipeline_btn.clicked.connect(self._on_run_pipeline)
        header.addWidget(self._run_pipeline_btn)
        self._rerun_pipeline_btn = QPushButton("Re-run Current")
        self._rerun_pipeline_btn.clicked.connect(self._on_rerun_pipeline)
        header.addWidget(self._rerun_pipeline_btn)
        self._run_all_missing_btn = QPushButton("Run All Missing")
        self._run_all_missing_btn.clicked.connect(self._on_run_all_missing)
        header.addWidget(self._run_all_missing_btn)
        self._sidebar_toggle_btn = QPushButton("◀")
        self._sidebar_toggle_btn.setToolTip("Toggle sidebar [Esc]")
        self._sidebar_toggle_btn.clicked.connect(self._on_toggle_sidebar)
        header.addWidget(self._sidebar_toggle_btn)
        self._metrics_label = QLabel("")
        self._metrics_label.setStyleSheet("font-size: 10px; color: #7d91a6;")
        header.addWidget(self._metrics_label)
        layout.addLayout(header)
        sep = QFrame()
        sep.setObjectName("headerSep")
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        layout.addWidget(sep)

        # Content: left (filters + queue table + meta + form), right (plot tabs)
        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter = self._main_splitter
        self._left_panel_width = 400

        # ---------- Left: filters + queue table + metadata + form ----------
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(14)

        # Full filter sidebar (Dash parity: SIDEBAR_GROUPS + bounds + sort_cols)
        self._filter_widget = FilterSidebarWidget()
        self._filter_widget.refresh_btn().clicked.connect(self._apply_filters_and_refresh_queue)
        self._filter_widget.bounds_btn().clicked.connect(self._on_refresh_filter_bounds)
        left_layout.addWidget(self._filter_widget)

        # Queue table
        self._queue_table = QTableWidget(0, 3)
        self._queue_table.setHorizontalHeaderLabels(["ID", "Path", "Status"])
        self._queue_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._queue_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._queue_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._queue_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._queue_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._queue_table.setAlternatingRowColors(True)
        self._queue_table.verticalHeader().setDefaultSectionSize(28)
        self._queue_table.currentCellChanged.connect(self._on_queue_row_changed)
        left_layout.addWidget(self._queue_table)

        # Vetting banner
        self._vetting_banner = QLabel("")
        self._vetting_banner.setWordWrap(True)
        self._vetting_banner.setStyleSheet("font-size: 10px; color: #7d91a6;")
        left_layout.addWidget(self._vetting_banner)

        # Run config panel
        run_config_group = QGroupBox("Run config")
        self._run_config_text = QPlainTextEdit()
        self._run_config_text.setReadOnly(True)
        self._run_config_text.setMaximumBlockCount(50)
        run_config_layout = QVBoxLayout(run_config_group)
        run_config_layout.addWidget(self._run_config_text)
        left_layout.addWidget(run_config_group)

        # Pipeline status chips + log (Dash parity)
        pipeline_group = QGroupBox("Pipeline")
        pipeline_layout = QVBoxLayout(pipeline_group)
        self._pipeline_chips = QLabel("—")
        self._pipeline_chips.setWordWrap(True)
        self._pipeline_chips.setStyleSheet("font-size: 10px;")
        pipeline_layout.addWidget(self._pipeline_chips)
        self._pipeline_log = QPlainTextEdit()
        self._pipeline_log.setReadOnly(True)
        self._pipeline_log.setMaximumBlockCount(500)
        self._pipeline_log.setMaximumHeight(120)
        pipeline_layout.addWidget(self._pipeline_log)
        left_layout.addWidget(pipeline_group)

        # Collapsible metadata (QTreeWidget with top-level groups); Vetting value cells are editable
        meta_group = QGroupBox("Metadata")
        self._meta_tree = QTreeWidget()
        self._meta_tree.setHeaderLabels(["Key", "Value"])
        self._meta_tree.setColumnCount(2)
        self._meta_tree.itemChanged.connect(self._on_meta_tree_item_changed)
        meta_layout = QVBoxLayout(meta_group)
        meta_layout.addWidget(self._meta_tree)
        left_layout.addWidget(meta_group)

        # Followup + Notes
        self._followup_check = QCheckBox("Needs follow-up")
        self._followup_check.stateChanged.connect(self._on_followup_changed)
        left_layout.addWidget(self._followup_check)
        left_layout.addWidget(QLabel("Notes:"))
        self._notes_edit = QPlainTextEdit()
        self._notes_edit.setPlaceholderText("Notes...")
        self._notes_edit.setMaximumBlockCount(200)
        left_layout.addWidget(self._notes_edit)

        # Review form
        form_frame = QFrame()
        form_frame.setObjectName("formFrame")
        form_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        form_layout = QVBoxLayout(form_frame)
        form_layout.addWidget(QLabel("Class (click or key):"))
        class_row = QHBoxLayout()
        self._class_buttons = {}
        for key, label in [
            ("d", "D"), ("l", "L"), ("m", "M"), ("f", "F"),
            ("u", "U"), ("i", "I"), ("o", "O"),
        ]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, k=key: self._set_class(k))
            self._class_buttons[key] = btn
            class_row.addWidget(btn)
        form_layout.addLayout(class_row)
        form_layout.addWidget(QLabel("Confidence (1–4):"))
        score_row = QHBoxLayout()
        self._score_buttons = {}
        for i in range(1, 5):
            btn = QPushButton(str(i))
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, n=i: self._set_score(n))
            self._score_buttons[i] = btn
            score_row.addWidget(btn)
        form_layout.addLayout(score_row)
        save_btn = QPushButton("Save (.)")
        save_btn.setObjectName("saveBtn")
        save_btn.clicked.connect(self._save_review)
        form_layout.addWidget(save_btn)
        left_layout.addWidget(form_frame)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left)
        left_scroll.setMaximumWidth(900)
        splitter.addWidget(left_scroll)

        # ---------- Right: plot tabs (Light curve + diagnostics) ----------
        self._plot_tab = QTabWidget()
        # Light curve tab: controls + plot (Dash parity: preset, overlays, bands, opacity, residual, export)
        self._lc_tab = QWidget()
        lc_layout = QVBoxLayout(self._lc_tab)
        lc_toolbar = QHBoxLayout()
        lc_toolbar.addWidget(QLabel("Preset:"))
        self._plot_preset = QComboBox()
        self._plot_preset.addItems(["Clean", "Diagnostics", "Full"])
        self._plot_preset.setCurrentText("Diagnostics")
        self._plot_preset.currentTextChanged.connect(self._on_plot_preset_changed)
        lc_toolbar.addWidget(self._plot_preset)
        lc_toolbar.addWidget(QLabel("Overlays:"))
        self._plot_overlay_raw = QCheckBox("raw")
        self._plot_overlay_markers = QCheckBox("markers")
        self._plot_overlay_residuals = QCheckBox("residuals")
        self._plot_overlay_phase = QCheckBox("phase")
        self._plot_overlay_filter_bad = QCheckBox("filter_bad")
        self._plot_overlay_diagnostics = QCheckBox("diagnostics")
        self._plot_overlay_confidence = QCheckBox("confidence")
        for cb in (self._plot_overlay_raw, self._plot_overlay_markers, self._plot_overlay_residuals,
                   self._plot_overlay_phase, self._plot_overlay_filter_bad, self._plot_overlay_diagnostics,
                   self._plot_overlay_confidence):
            cb.stateChanged.connect(lambda: self._refresh_current_candidate())
        # Defaults for Diagnostics preset
        for cb in (self._plot_overlay_raw, self._plot_overlay_markers, self._plot_overlay_residuals,
                   self._plot_overlay_phase, self._plot_overlay_filter_bad, self._plot_overlay_diagnostics):
            cb.setChecked(True)
        lc_toolbar.addWidget(self._plot_overlay_raw)
        lc_toolbar.addWidget(self._plot_overlay_markers)
        lc_toolbar.addWidget(self._plot_overlay_residuals)
        lc_toolbar.addWidget(self._plot_overlay_phase)
        lc_toolbar.addWidget(self._plot_overlay_filter_bad)
        lc_toolbar.addWidget(self._plot_overlay_diagnostics)
        lc_toolbar.addWidget(self._plot_overlay_confidence)
        lc_toolbar.addWidget(QLabel("Bands:"))
        self._plot_band_g = QCheckBox("g")
        self._plot_band_V = QCheckBox("V")
        self._plot_band_g.setChecked(True)
        self._plot_band_V.setChecked(True)
        self._plot_band_g.stateChanged.connect(lambda: self._refresh_current_candidate())
        self._plot_band_V.stateChanged.connect(lambda: self._refresh_current_candidate())
        lc_toolbar.addWidget(self._plot_band_g)
        lc_toolbar.addWidget(self._plot_band_V)
        lc_toolbar.addWidget(QLabel("Baseline opacity:"))
        self._plot_baseline_opacity = QSlider(Qt.Orientation.Horizontal)
        self._plot_baseline_opacity.setRange(0, 100)
        self._plot_baseline_opacity.setValue(50)
        self._plot_baseline_opacity.setMaximumWidth(80)
        self._plot_baseline_opacity.valueChanged.connect(lambda: self._refresh_current_candidate())
        lc_toolbar.addWidget(self._plot_baseline_opacity)
        lc_toolbar.addWidget(QLabel("Residual:"))
        self._plot_residual_frac = QSlider(Qt.Orientation.Horizontal)
        self._plot_residual_frac.setRange(5, 50)
        self._plot_residual_frac.setValue(25)
        self._plot_residual_frac.setMaximumWidth(60)
        self._plot_residual_frac.valueChanged.connect(lambda: self._refresh_current_candidate())
        lc_toolbar.addWidget(self._plot_residual_frac)
        self._plot_export_btn = QPushButton("Export plot")
        self._plot_export_btn.clicked.connect(self._on_export_plot)
        lc_toolbar.addWidget(self._plot_export_btn)
        lc_toolbar.addStretch()
        lc_layout.addLayout(lc_toolbar)
        if _HAS_WEBENGINE and QWebEngineView is not None:
            self._plot_view = QWebEngineView()
            self._plot_image_label = None
        else:
            self._plot_view = None
            self._plot_image_label = QLabel("Install PySide6-WebEngine for interactive plots.")
            self._plot_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._plot_image_label.setScaledContents(True)
        if self._plot_view is not None:
            self._plot_view.setMinimumSize(400, 300)
            lc_layout.addWidget(self._plot_view, 1)  # stretch so plot fills tab in fullscreen
        else:
            self._plot_image_label.setMinimumSize(400, 300)
            lc_layout.addWidget(self._plot_image_label, 1)
        self._plot_tab.addTab(self._lc_tab, "Light curve")

        # Diagnostic tabs: one widget per tab (WebEngine or QLabel), created on first switch
        self._diagnostic_tab_widgets: list[QWidget] = []
        for name, _ in DIAGNOSTIC_BUILDERS:
            placeholder = QLabel(f"Select a candidate and switch to this tab to load {name}.")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._diagnostic_tab_widgets.append(placeholder)
            self._plot_tab.addTab(placeholder, name)
        self._plot_tab.currentChanged.connect(self._on_plot_tab_changed)
        splitter.addWidget(self._plot_tab)
        splitter.setSizes([400, 800])
        layout.addWidget(splitter, 1)  # stretch so content fills window when maximized/fullscreen

        self._status = QStatusBar()
        self.setStatusBar(self._status)

    def _connect_db(self) -> None:
        if not self._db_path.exists():
            self._status.showMessage(f"DB not found: {self._db_path}")
            return
        self._conn = db_connect(self._db_path)

    def _build_filter_params(self) -> dict:
        return self._filter_widget.get_filter_params()

    def _load_filter_persistence(self) -> None:
        if self._conn is None:
            return
        state = load_app_state(self._conn, "qt_filter_params", "")
        if state.strip():
            try:
                self._filter_widget.set_filter_params(json.loads(state))
            except Exception:
                pass

    def _on_refresh_filter_bounds(self) -> None:
        if self._conn is None:
            self._filter_widget.set_bounds_status("No DB connected.")
            return
        self._filter_widget.bounds_btn().setEnabled(False)
        self._filter_widget.set_bounds_status("Loading slider bounds...")
        try:
            self._filter_widget.load_numeric_bounds_sync(self._conn)
            self._filter_widget.populate_options(self._conn)
            self._filter_widget.set_bounds_status("Bounds and options loaded.")
        except Exception as e:
            self._filter_widget.set_bounds_status(f"Error: {e}")
        finally:
            self._filter_widget.bounds_btn().setEnabled(True)

    def _load_queue(self) -> None:
        if self._conn is None:
            return
        self._filter_params = self._build_filter_params()
        df = query_queue(self._conn, filters=self._filter_params, ids_only=False)
        self._queue_df = df
        self._candidate_ids = df["candidate_id"].tolist() if not df.empty else []
        self._current_idx = 0
        self._populate_queue_table()
        self._sync_queue_table_selection()
        self._update_progress_label()

    def _populate_queue_table(self) -> None:
        self._queue_table.blockSignals(True)
        self._queue_table.setRowCount(len(self._candidate_ids))
        for i, cid in enumerate(self._candidate_ids):
            self._queue_table.setItem(i, 0, QTableWidgetItem(str(cid)))
            path = ""
            status = ""
            if not self._queue_df.empty and i < len(self._queue_df):
                row = self._queue_df.iloc[i]
                path = str(row.get("lc_path", "") or "")[:60]
                status = str(row.get("status", "") or "")
            self._queue_table.setItem(i, 1, QTableWidgetItem(path))
            self._queue_table.setItem(i, 2, QTableWidgetItem(status))
        self._queue_table.blockSignals(False)

    def _sync_queue_table_selection(self) -> None:
        if not self._candidate_ids:
            return
        if self._current_idx >= len(self._candidate_ids):
            self._current_idx = len(self._candidate_ids) - 1
        self._queue_table.blockSignals(True)
        self._queue_table.setCurrentCell(self._current_idx, 0)
        item = self._queue_table.item(self._current_idx, 0)
        if item is not None:
            self._queue_table.scrollToItem(item)
        self._queue_table.blockSignals(False)

    def _on_queue_row_changed(self, current_row: int, current_col: int, prev_row: int, prev_col: int) -> None:
        if current_row < 0 or current_row >= len(self._candidate_ids):
            return
        if current_row == self._current_idx:
            return
        self._current_idx = current_row
        self._refresh_current_candidate()

    def _apply_filters_and_refresh_queue(self) -> None:
        self._load_queue()
        self._refresh_current_candidate()
        if self._conn is not None:
            try:
                save_app_state(self._conn, "qt_filter_params", json.dumps(self._filter_widget.get_filter_params(), default=str))
            except Exception:
                pass
        self._status.showMessage("Queue refreshed.", 2000)

    def _apply_theme_stylesheet(self) -> None:
        if self._theme == "black":
            sheet = """
                QMainWindow, QWidget { background-color: #1a1d21; color: #e4e6eb; font-size: 13px; }
                QGroupBox { font-weight: 600; margin-top: 12px; padding: 12px; padding-top: 20px; border: 1px solid #2d3238; border-radius: 8px; background-color: #21262d; }
                QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 8px; color: #8b949e; font-weight: 600; }
                QFrame#formFrame { background-color: #21262d; border: 1px solid #2d3238; border-radius: 8px; padding: 12px; }
                QPushButton { padding: 8px 14px; border-radius: 8px; border: 1px solid #3d434a; background: #21262d; min-width: 60px; min-height: 20px; }
                QPushButton:hover { background: #2d333b; border-color: #58a6ff; }
                QPushButton:pressed { background: #161b22; }
                QPushButton:checked { background: #238636; border-color: #2ea043; color: #fff; }
                QPushButton#saveBtn, QPushButton#runPipelineBtn { background: #238636; border-color: #2ea043; color: #fff; }
                QPushButton#saveBtn:hover, QPushButton#runPipelineBtn:hover { background: #2ea043; border-color: #3fb950; }
                QTableWidget { gridline-color: transparent; background: #21262d; alternate-background-color: #25282e; }
                QTableWidget::item { padding: 6px 10px; }
                QTableWidget::item:hover { background: #2d333b; }
                QHeaderView::section { padding: 8px 10px; background: #21262d; border: none; border-bottom: 2px solid #58a6ff; border-right: none; font-weight: 600; }
                QComboBox { padding: 8px; border-radius: 6px; border: 1px solid #3d434a; background: #21262d; min-width: 80px; }
                QComboBox:focus { border-color: #58a6ff; }
                QPlainTextEdit, QTextEdit { padding: 8px; border-radius: 6px; border: 1px solid #2d3238; background: #0d1117; }
                QPlainTextEdit:focus, QTextEdit:focus { border-color: #58a6ff; }
                QTreeWidget { border: 1px solid #2d3238; border-radius: 8px; background: #21262d; }
                QTabWidget::pane { border: 1px solid #2d3238; border-top: none; border-radius: 0 0 8px 8px; top: -1px; background: #21262d; }
                QTabBar::tab { padding: 10px 16px; margin-right: 4px; background: transparent; border: none; border-bottom: 2px solid transparent; color: #8b949e; }
                QTabBar::tab:selected { color: #e4e6eb; border-bottom: 2px solid #58a6ff; background: #21262d; }
                QTabBar::tab:selected:hover { color: #e4e6eb; border-bottom: 2px solid #58a6ff; background: #21262d; }
                QTabBar::tab:hover { color: #e4e6eb; }
                QScrollArea { border: none; }
                QFrame { border-radius: 8px; }
                QCheckBox { spacing: 8px; }
                QLabel { color: #e4e6eb; font-weight: 500; }
                QLabel#candidateLabel { color: #58a6ff; }
                QFrame#headerSep { background-color: #2d3238; border: none; }
                QMenuBar { background: #21262d; padding: 4px 0; }
                QMenuBar::item:selected { background: #2d333b; border-radius: 4px; }
                QStatusBar { background: #21262d; border-top: 1px solid #2d3238; font-size: 12px; }
            """
        elif self._theme == "white":
            sheet = """
                QMainWindow, QWidget { background-color: #f6f8fa; color: #1f2328; font-size: 13px; }
                QGroupBox { font-weight: 600; margin-top: 12px; padding: 12px; padding-top: 20px; border: 1px solid #d0d7de; border-radius: 8px; background-color: #ffffff; }
                QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 8px; color: #57606a; font-weight: 600; }
                QFrame#formFrame { background-color: #ffffff; border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; }
                QPushButton { padding: 8px 14px; border-radius: 8px; border: 1px solid #d0d7de; background: #f6f8fa; min-width: 60px; min-height: 20px; }
                QPushButton:hover { background: #eaeef2; border-color: #0969da; }
                QPushButton:pressed { background: #ddf4ff; }
                QPushButton:checked { background: #0969da; border-color: #0550ae; color: #fff; }
                QPushButton#saveBtn, QPushButton#runPipelineBtn { background: #0969da; border-color: #0550ae; color: #fff; }
                QPushButton#saveBtn:hover, QPushButton#runPipelineBtn:hover { background: #218bff; border-color: #0969da; }
                QTableWidget { gridline-color: transparent; background: #ffffff; alternate-background-color: #f6f8fa; }
                QTableWidget::item { padding: 6px 10px; }
                QTableWidget::item:hover { background: #eaeef2; }
                QHeaderView::section { padding: 8px 10px; background: #f6f8fa; border: none; border-bottom: 2px solid #0969da; border-right: none; font-weight: 600; }
                QComboBox { padding: 8px; border-radius: 6px; border: 1px solid #d0d7de; background: #fff; min-width: 80px; }
                QComboBox:focus { border-color: #0969da; }
                QPlainTextEdit, QTextEdit { padding: 8px; border-radius: 6px; border: 1px solid #d0d7de; background: #fff; }
                QPlainTextEdit:focus, QTextEdit:focus { border-color: #0969da; }
                QTreeWidget { border: 1px solid #d0d7de; border-radius: 8px; background: #fff; }
                QTabWidget::pane { border: 1px solid #d0d7de; border-top: none; border-radius: 0 0 8px 8px; top: -1px; background: #fff; }
                QTabBar::tab { padding: 10px 16px; margin-right: 4px; background: transparent; border: none; border-bottom: 2px solid transparent; color: #57606a; }
                QTabBar::tab:selected { color: #1f2328; border-bottom: 2px solid #0969da; background: #fff; }
                QTabBar::tab:selected:hover { color: #1f2328; border-bottom: 2px solid #0969da; background: #fff; }
                QTabBar::tab:hover { color: #1f2328; }
                QScrollArea { border: none; }
                QFrame { border-radius: 8px; }
                QCheckBox { spacing: 8px; }
                QLabel { color: #1f2328; font-weight: 500; }
                QLabel#candidateLabel { color: #0969da; }
                QFrame#headerSep { background-color: #d0d7de; border: none; }
                QMenuBar { background: #ffffff; padding: 4px 0; border-bottom: 1px solid #d0d7de; }
                QMenuBar::item:selected { background: #eaeef2; border-radius: 4px; }
                QStatusBar { background: #ffffff; border-top: 1px solid #d0d7de; font-size: 12px; }
            """
        else:
            sheet = """
                QMainWindow, QWidget { background-color: #1a1d21; color: #e4e6eb; font-size: 13px; }
                QGroupBox { font-weight: 600; margin-top: 12px; padding: 12px; padding-top: 20px; border: 1px solid #2d3238; border-radius: 8px; background-color: #21262d; }
                QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 8px; color: #8b949e; font-weight: 600; }
                QFrame#formFrame { background-color: #21262d; border: 1px solid #2d3238; border-radius: 8px; padding: 12px; }
                QPushButton { padding: 8px 14px; border-radius: 8px; border: 1px solid #3d434a; background: #21262d; min-width: 60px; min-height: 20px; }
                QPushButton:hover { background: #2d333b; border-color: #58a6ff; }
                QPushButton:pressed { background: #161b22; }
                QPushButton:checked { background: #238636; border-color: #2ea043; color: #fff; }
                QPushButton#saveBtn, QPushButton#runPipelineBtn { background: #238636; border-color: #2ea043; color: #fff; }
                QPushButton#saveBtn:hover, QPushButton#runPipelineBtn:hover { background: #2ea043; border-color: #3fb950; }
                QTableWidget { gridline-color: transparent; background: #21262d; alternate-background-color: #25282e; }
                QTableWidget::item { padding: 6px 10px; }
                QTableWidget::item:hover { background: #2d333b; }
                QHeaderView::section { padding: 8px 10px; background: #21262d; border: none; border-bottom: 2px solid #58a6ff; border-right: none; font-weight: 600; }
                QComboBox { padding: 8px; border-radius: 6px; border: 1px solid #3d434a; background: #21262d; min-width: 80px; }
                QComboBox:focus { border-color: #58a6ff; }
                QPlainTextEdit, QTextEdit { padding: 8px; border-radius: 6px; border: 1px solid #2d3238; background: #0d1117; }
                QPlainTextEdit:focus, QTextEdit:focus { border-color: #58a6ff; }
                QTreeWidget { border: 1px solid #2d3238; border-radius: 8px; background: #21262d; }
                QTabWidget::pane { border: 1px solid #2d3238; border-top: none; border-radius: 0 0 8px 8px; top: -1px; background: #21262d; }
                QTabBar::tab { padding: 10px 16px; margin-right: 4px; background: transparent; border: none; border-bottom: 2px solid transparent; color: #8b949e; }
                QTabBar::tab:selected { color: #e4e6eb; border-bottom: 2px solid #58a6ff; background: #21262d; }
                QTabBar::tab:selected:hover { color: #e4e6eb; border-bottom: 2px solid #58a6ff; background: #21262d; }
                QTabBar::tab:hover { color: #e4e6eb; }
                QScrollArea { border: none; }
                QFrame { border-radius: 8px; }
                QCheckBox { spacing: 8px; }
                QLabel { color: #e4e6eb; font-weight: 500; }
                QLabel#candidateLabel { color: #58a6ff; }
                QFrame#headerSep { background-color: #2d3238; border: none; }
                QMenuBar { background: #21262d; padding: 4px 0; }
                QMenuBar::item:selected { background: #2d333b; border-radius: 4px; }
                QStatusBar { background: #21262d; border-top: 1px solid #2d3238; font-size: 12px; }
            """
        self.setStyleSheet(sheet)

    def _on_theme_changed(self, text: str) -> None:
        m = {"Black": "black", "White": "white"}
        self._theme = m.get(text, "black")
        self._apply_theme_stylesheet()
        self._refresh_current_candidate()

    def _on_followup_changed(self, state: int) -> None:
        self._needs_followup = (state == Qt.CheckState.Checked)

    def _update_progress_label(self) -> None:
        n = len(self._candidate_ids)
        if n == 0:
            self._progress_label.setText("[0/0]")
            return
        self._progress_label.setText(f"[{self._current_idx + 1}/{n}]")

    def _current_candidate_id(self) -> str | None:
        if not self._candidate_ids or self._current_idx < 0 or self._current_idx >= len(self._candidate_ids):
            return None
        return self._candidate_ids[self._current_idx]

    def _refresh_current_candidate(self) -> None:
        self._sync_queue_table_selection()
        cid = self._current_candidate_id()
        if cid is None:
            self._candidate_label.setText("—")
            self._vetting_banner.setText("")
            self._run_config_text.setPlainText("")
            self._meta_tree.clear()
            self._followup_check.setChecked(False)
            self._notes_edit.setPlainText("")
            pv = getattr(self, "_plot_view", None)
            pil = getattr(self, "_plot_image_label", None)
            if pv is not None:
                pv.setHtml("<p>No candidate selected.</p>")
            elif pil is not None:
                pil.setText("No candidate selected.")
            self._update_class_score_buttons()
            return
        self._candidate_label.setText(str(cid))
        self._update_progress_label()
        if self._conn is None:
            return
        payload = get_candidate_payload(self._conn, cid)
        review = get_review(self._conn, cid)
        self._current_score = review.get("interest_score")
        self._current_class = review.get("event_class") or "unclassified"
        self._review_pass = review.get("review_pass") or 1
        self._needs_followup = (review.get("status") or "") == "needs_followup"
        self._followup_check.blockSignals(True)
        self._followup_check.setChecked(self._needs_followup)
        self._followup_check.blockSignals(False)
        self._notes_edit.setPlainText(review.get("notes") or "")

        vet_parts = []
        for k in ("vetting_likely_known", "simbad_otype", "gaia_var_class", "asassn_var_type", "ztf_var_type"):
            v = payload.get(k)
            if v is not None and str(v).strip():
                vet_parts.append(f"{k}: {v}")
        self._vetting_banner.setText(" | ".join(vet_parts) if vet_parts else "No vetting summary.")

        run_params = payload.get("run_params") or {}
        rp_lines = [f"{k}: {v}" for k, v in sorted(run_params.items())]
        self._run_config_text.setPlainText("\n".join(rp_lines) if rp_lines else "No run config.")

        try:
            from malca.review.pipeline import detect_pipeline_status
            status = detect_pipeline_status(payload)
            chips = " ".join(f"{s}: {v}" for s, v in sorted(status.items()))
            self._pipeline_chips.setText(chips or "—")
        except Exception:
            self._pipeline_chips.setText("—")

        self._meta_tree.blockSignals(True)
        try:
            self._meta_tree.clear()
            _EDITABLE_META_KEYS = frozenset({
                "vetting_likely_known", "simbad_otype", "gaia_var_class",
                "asassn_var_type", "ztf_var_type", "vsx_class", "tns_type",
            })
            sections = [
                ("IDs", ["candidate_id", "asas_sn_id", "gaia_id", "simbad_main_id"]),
                ("Paths", ["path", "dat_path", "lc_path", "source_path"]),
                ("Vetting", [
                    "vetting_likely_known", "simbad_otype", "gaia_var_class",
                    "asassn_var_type", "ztf_var_type", "vsx_class", "tns_type",
                ]),
                ("Flags", ["failed_any", "catalog_match", "high_ruwe_flag", "periodic_flag"]),
            ]
            for section_name, keys in sections:
                root = QTreeWidgetItem(self._meta_tree, [section_name, ""])
                for k in keys:
                    v = payload.get(k)
                    if v is not None:
                        item = QTreeWidgetItem(root, [k, str(v)])
                    else:
                        item = QTreeWidgetItem(root, [k, ""])
                    if k in _EDITABLE_META_KEYS:
                        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                        item.setData(0, Qt.ItemDataRole.UserRole, k)
                root.setExpanded(False)
        finally:
            self._meta_tree.blockSignals(False)

        self._update_class_score_buttons()

        filter_bad = getattr(self, "_plot_overlay_filter_bad", None) and self._plot_overlay_filter_bad.isChecked()
        show_diag = getattr(self, "_plot_overlay_diagnostics", None) and self._plot_overlay_diagnostics.isChecked()
        confidence = getattr(self, "_plot_overlay_confidence", None) and self._plot_overlay_confidence.isChecked()
        baseline_opacity = 0.5
        residual_fraction = 0.25
        selected_bands = ["g", "V"]
        if getattr(self, "_plot_baseline_opacity", None):
            baseline_opacity = self._plot_baseline_opacity.value() / 100.0
        if getattr(self, "_plot_residual_frac", None):
            residual_fraction = self._plot_residual_frac.value() / 100.0
        if getattr(self, "_plot_band_g", None) and getattr(self, "_plot_band_V", None):
            selected_bands = []
            if self._plot_band_g.isChecked():
                selected_bands.append("g")
            if self._plot_band_V.isChecked():
                selected_bands.append("V")
            if not selected_bands:
                selected_bands = ["g", "V"]
        result = build_interactive_lightcurve_figure(
            payload,
            plot_dir=self._plot_dir,
            selected_cameras=None,
            filter_bad_cameras=filter_bad,
            show_baseline=True,
            show_event_markers=getattr(self, "_plot_overlay_markers", None) and self._plot_overlay_markers.isChecked(),
            show_residuals=getattr(self, "_plot_overlay_residuals", None) and self._plot_overlay_residuals.isChecked(),
            show_phase_fold=getattr(self, "_plot_overlay_phase", None) and self._plot_overlay_phase.isChecked(),
            show_raw_mag=getattr(self, "_plot_overlay_raw", None) and self._plot_overlay_raw.isChecked(),
            show_diagnostics=show_diag,
            confidence_colors=confidence,
            run_params=run_params,
            uirevision_key=cid,
            theme=self._theme,
            residual_fraction=residual_fraction,
            baseline_opacity=baseline_opacity,
            selected_bands=selected_bands or None,
        )
        fig = result.get("figure")
        self._last_lc_fig = fig
        pv = getattr(self, "_plot_view", None)
        pil = getattr(self, "_plot_image_label", None)
        if fig is not None:
            if pv is not None:
                html = fig.to_html(full_html=True, include_plotlyjs="cdn")
                pv.setHtml(html)
            elif pil is not None:
                try:
                    img_bytes = fig.to_image(format="png", width=800, height=500, scale=2)
                    qimg = QImage()
                    qimg.loadFromData(img_bytes)
                    pil.setPixmap(QPixmap.fromImage(qimg))
                except Exception:
                    pil.setText("Plot failed (install kaleido for image export).")
        else:
            if pv is not None:
                pv.setHtml("<p>No figure.</p>")
            elif pil is not None:
                pil.setText("No figure.")

        idx = self._plot_tab.currentIndex()
        if idx > 0:
            self._render_diagnostic_tab(idx)

    def _on_plot_preset_changed(self, preset: str) -> None:
        overlays = {"Clean": ["raw", "markers", "residuals", "phase", "filter_bad_cameras"],
                    "Diagnostics": ["raw", "markers", "residuals", "phase", "filter_bad_cameras", "diagnostics"],
                    "Full": ["raw", "markers", "residuals", "phase", "filter_bad_cameras", "diagnostics", "confidence"]}
        want = set(overlays.get(preset, overlays["Diagnostics"]))
        self._plot_overlay_raw.setChecked("raw" in want)
        self._plot_overlay_markers.setChecked("markers" in want)
        self._plot_overlay_residuals.setChecked("residuals" in want)
        self._plot_overlay_phase.setChecked("phase" in want)
        self._plot_overlay_filter_bad.setChecked("filter_bad_cameras" in want)
        self._plot_overlay_diagnostics.setChecked("diagnostics" in want)
        self._plot_overlay_confidence.setChecked("confidence" in want)
        self._refresh_current_candidate()

    def _on_export_plot(self) -> None:
        if self._last_lc_fig is None:
            self._status.showMessage("No plot to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export plot", "", "PNG (*.png);;SVG (*.svg);;HTML (*.html)"
        )
        if not path:
            return
        try:
            if path.lower().endswith(".html"):
                with open(path, "w") as f:
                    f.write(self._last_lc_fig.to_html(full_html=True, include_plotlyjs="cdn"))
            elif path.lower().endswith(".svg"):
                self._last_lc_fig.write_image(path, format="svg")
            else:
                self._last_lc_fig.write_image(path, format="png", width=1200, height=700, scale=2)
            self._status.showMessage(f"Saved: {path}", 3000)
        except Exception as e:
            self._status.showMessage(f"Export failed: {e}")
            QMessageBox.warning(self, "Export", str(e))

    def _on_plot_tab_changed(self, index: int) -> None:
        if index > 0:
            self._render_diagnostic_tab(index)

    def _get_diagnostic_background(self) -> dict:
        if self._diagnostic_background is None and self._conn is not None:
            try:
                self._diagnostic_background = get_diagnostic_background(self._conn)
            except Exception:
                self._diagnostic_background = {}
        return self._diagnostic_background or {}

    def _render_diagnostic_tab(self, tab_index: int) -> None:
        if tab_index <= 0 or tab_index > len(DIAGNOSTIC_BUILDERS):
            return
        cid = self._current_candidate_id()
        if cid is None or self._conn is None:
            return
        payload = get_candidate_payload(self._conn, cid)
        name, builder = DIAGNOSTIC_BUILDERS[tab_index - 1]
        if builder is None:
            return
        background = self._get_diagnostic_background()
        try:
            fig = builder(payload, self._theme, background=background)
        except Exception:
            fig = None
        placeholder = self._diagnostic_tab_widgets[tab_index - 1]
        if fig is not None:
            html = fig.to_html(full_html=True, include_plotlyjs="cdn")
            if _HAS_WEBENGINE and QWebEngineView is not None and not hasattr(placeholder, "setHtml"):
                view = QWebEngineView()
                view.setHtml(html)
                view.setMinimumSize(400, 300)
                self._plot_tab.removeTab(tab_index)
                self._plot_tab.insertTab(tab_index, view, name)
                self._diagnostic_tab_widgets[tab_index - 1] = view
            elif hasattr(placeholder, "setHtml"):
                placeholder.setHtml(html)
            else:
                try:
                    img_bytes = fig.to_image(format="png", width=700, height=400, scale=2)
                    qimg = QImage()
                    qimg.loadFromData(img_bytes)
                    placeholder.setPixmap(QPixmap.fromImage(qimg))
                    placeholder.setScaledContents(True)
                except Exception:
                    placeholder.setText(f"Failed to render {name}.")
        else:
            if QWebEngineView is None or not isinstance(placeholder, QWebEngineView):
                if hasattr(placeholder, "setText"):
                    placeholder.setText(f"No data for {name}.")

    def _update_class_score_buttons(self) -> None:
        for key, btn in self._class_buttons.items():
            btn.setChecked(CLASS_KEY_MAP.get(key) == self._current_class)
        for i, btn in self._score_buttons.items():
            btn.setChecked(self._current_score == i)

    def _set_class(self, key: str) -> None:
        cls = CLASS_KEY_MAP.get(key, "unclassified")
        if cls == self._current_class:
            self._current_class = "unclassified"
        else:
            self._current_class = cls
        self._update_class_score_buttons()
        self._status.showMessage(f"Class: {self._current_class}", 2000)

    def _set_score(self, score: int) -> None:
        self._current_score = score
        self._update_class_score_buttons()
        self._status.showMessage(f"Confidence: {score}", 2000)

    def _save_review(self) -> None:
        cid = self._current_candidate_id()
        if cid is None or self._conn is None:
            self._status.showMessage("No candidate to save.")
            return
        status = "needs_followup" if self._needs_followup else "reviewed"
        notes = self._notes_edit.toPlainText().strip()
        save_review(
            self._conn,
            candidate_id=cid,
            interest_score=self._current_score,
            event_class=self._current_class,
            review_pass=self._review_pass,
            notes=notes,
            status=status,
            reviewer="qt",
        )
        self._status.showMessage("Saved.", 2000)

    def _on_meta_tree_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if column != 1 or self._conn is None:
            return
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if not key:
            return
        value = item.text(1).strip()
        cid = self._current_candidate_id()
        if not cid:
            return
        if key == "vetting_likely_known":
            value = 1 if value and value.lower() in ("1", "true", "yes") else 0
        ok = replace_candidate_payload_fields(self._conn, cid, {key: value})
        if ok:
            self._status.showMessage(f"Saved {key}.", 2000)

    def _install_shortcuts(self) -> None:
        # Qt key names for special keys
        key_map = {"Enter": "Return"}
        # Navigation and actions
        for key, action in KEYBOARD_SHORTCUTS.items():
            if action == "show_shortcuts":
                continue
            if action == "toggle_sidebar":
                QShortcut(QKeySequence("Escape"), self).activated.connect(self._on_toggle_sidebar)
                continue
            qkey = key_map.get(key, key)
            try:
                seq = QKeySequence(qkey)
            except Exception:
                seq = QKeySequence(key)
            shortcut = QShortcut(seq, self)
            shortcut.activated.connect(lambda k=key: self._on_key(k))
        # Class keys
        for key in CLASS_KEY_MAP:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda k=key: self._set_class(k))
        # Help
        QShortcut(QKeySequence("?"), self).activated.connect(self._show_help)

    def _on_key(self, key: str) -> None:
        cid = self._current_candidate_id()
        queue_size = len(self._candidate_ids)
        new_idx, msg, should_save = handle_key_action(
            key, self._current_idx, queue_size, self._conn, cid
        )
        if should_save and cid and self._conn:
            status = "needs_followup" if self._needs_followup else "reviewed"
            notes = self._notes_edit.toPlainText().strip()
            save_review(
                self._conn,
                candidate_id=cid,
                interest_score=self._current_score,
                event_class=self._current_class,
                review_pass=self._review_pass,
                notes=notes,
                status=status,
                reviewer="qt",
            )
        self._current_idx = new_idx
        self._status.showMessage(msg, 2000)
        self._refresh_current_candidate()

    def _show_help(self) -> None:
        QMessageBox.information(self, "Shortcuts", HELP_TEXT)

    # ---------- Phase 4: Import / Export / Merge / Pipeline ----------
    def _on_import_candidates(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import candidates", "", "CSV (*.csv);;Parquet (*.parquet *.pq);;All (*)"
        )
        if not path:
            return
        try:
            if path.lower().endswith(".parquet") or path.lower().endswith(".pq"):
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.warning(self, "Import failed", f"Could not read file: {e}")
            return
        if df.empty:
            QMessageBox.warning(self, "Import failed", "File is empty.")
            return
        source_path, ok = QInputDialog.getText(self, "Source path", "Enter source_path for this import:", text="")
        if not ok or source_path is None:
            return
        source_path = str(source_path).strip() or "import"
        if self._conn is None:
            QMessageBox.warning(self, "Import failed", "Database not connected.")
            return
        try:
            n, _ = import_candidates(
                self._conn,
                df,
                source_path,
                characterize_before_import=False,
                vet_before_import=False,
            )
        except Exception as e:
            QMessageBox.warning(self, "Import failed", str(e))
            return
        self._diagnostic_background = None
        self._load_queue()
        self._refresh_current_candidate()
        QMessageBox.information(self, "Import", f"Imported {n} candidates.")

    def _on_export_reviews(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export reviews", "", "CSV (*.csv);;Parquet (*.parquet);;All (*)"
        )
        if not path:
            return
        if self._conn is None:
            QMessageBox.warning(self, "Export failed", "Database not connected.")
            return
        try:
            export_reviews(self._conn, Path(path), only_reviewed=True)
        except Exception as e:
            QMessageBox.warning(self, "Export failed", str(e))
            return
        QMessageBox.information(self, "Export", "Reviews exported.")

    def _on_merge_vetting(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Merge vetting results", "", "CSV (*.csv);;Parquet (*.parquet *.pq);;All (*)"
        )
        if not path:
            return
        try:
            if path.lower().endswith(".parquet") or path.lower().endswith(".pq"):
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.warning(self, "Merge failed", f"Could not read file: {e}")
            return
        if df.empty:
            QMessageBox.warning(self, "Merge failed", "File is empty.")
            return
        if self._conn is None:
            QMessageBox.warning(self, "Merge failed", "Database not connected.")
            return
        try:
            n = merge_vetting_results(self._conn, df, id_column=None)
        except Exception as e:
            QMessageBox.warning(self, "Merge failed", str(e))
            return
        self._diagnostic_background = None
        self._refresh_current_candidate()
        QMessageBox.information(self, "Merge vetting", f"Merged vetting for {n} candidates.")

    def _on_run_pipeline(self) -> None:
        self._run_pipeline_worker(force_rerun=False, run_all_missing=False)

    def _on_rerun_pipeline(self) -> None:
        self._run_pipeline_worker(force_rerun=True, run_all_missing=False)

    def _on_run_all_missing(self) -> None:
        if not self._candidate_ids:
            self._status.showMessage("Queue is empty.")
            return
        self._run_pipeline_worker(force_rerun=False, run_all_missing=True, candidate_ids=list(self._candidate_ids))

    def _run_pipeline_worker(self, force_rerun: bool = False, run_all_missing: bool = False, candidate_ids: list[str] | None = None) -> None:
        cid = self._current_candidate_id()
        if not cid or self._conn is None:
            self._status.showMessage("No candidate selected or DB not connected.")
            return
        self._run_pipeline_btn.setEnabled(False)
        self._rerun_pipeline_btn.setEnabled(False)
        self._run_all_missing_btn.setEnabled(False)
        self._pipeline_log.clear()
        self._status.showMessage("Running pipeline...")
        worker = PipelineWorker(self._db_path, cid, force_rerun=force_rerun, run_all_missing=run_all_missing, candidate_ids=candidate_ids or [])
        worker.progress.connect(self._on_pipeline_progress)
        worker.finished.connect(lambda: self._on_pipeline_finished(worker))
        worker.start()

    def _on_pipeline_progress(self, msg: str) -> None:
        self._pipeline_log.appendPlainText(msg)

    def _on_pipeline_finished(self, worker: "PipelineWorker") -> None:
        self._run_pipeline_btn.setEnabled(True)
        self._rerun_pipeline_btn.setEnabled(True)
        self._run_all_missing_btn.setEnabled(True)
        if worker.error:
            self._status.showMessage(f"Pipeline error: {worker.error}")
            self._pipeline_log.appendPlainText(f"Error: {worker.error}")
            QMessageBox.warning(self, "Pipeline", worker.error)
        else:
            self._status.showMessage("Pipeline finished.")
            self._diagnostic_background = None
            self._refresh_current_candidate()

    def _on_toggle_sidebar(self) -> None:
        if self._sidebar_collapsed:
            self._main_splitter.setSizes([self._left_panel_width, 800])
            self._sidebar_toggle_btn.setText("◀")
        else:
            self._left_panel_width = self._main_splitter.sizes()[0] or 400
            self._main_splitter.setSizes([0, 1200])
            self._sidebar_toggle_btn.setText("▶")
        self._sidebar_collapsed = not self._sidebar_collapsed

    def _update_metrics(self) -> None:
        n = len(self._candidate_ids)
        i = self._current_idx + 1 if n else 0
        p = self._review_pass
        self._metrics_label.setText(f"Queue: {n} | Pos: {i} | Pass: {p}")

    def _on_download_run_config(self) -> None:
        cid = self._current_candidate_id()
        if cid is None or self._conn is None:
            self._status.showMessage("No candidate selected.")
            return
        payload = get_candidate_payload(self._conn, cid)
        run_params = payload.get("run_params") or {}
        if not run_params:
            QMessageBox.information(self, "Run config", "No run_params for this candidate.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save run config", "run_params.json", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(run_params, f, indent=2, default=str)
            self._status.showMessage(f"Saved: {path}", 3000)
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))

    def _on_cone_search(self) -> None:
        cid = self._current_candidate_id()
        ra, dec = None, None
        if cid and self._conn:
            payload = get_candidate_payload(self._conn, cid)
            ra = payload.get("ra_deg")
            dec = payload.get("dec_deg")
        dlg = QDialog(self)
        dlg.setWindowTitle("Cone search")
        form = QFormLayout(dlg)
        ra_edit = QPlainTextEdit()
        ra_edit.setMaximumHeight(28)
        ra_edit.setPlainText(str(ra) if ra is not None else "")
        form.addRow("RA (deg):", ra_edit)
        dec_edit = QPlainTextEdit()
        dec_edit.setMaximumHeight(28)
        dec_edit.setPlainText(str(dec) if dec is not None else "")
        form.addRow("Dec (deg):", dec_edit)
        radius_edit = QPlainTextEdit()
        radius_edit.setMaximumHeight(28)
        radius_edit.setPlainText("5.0")
        form.addRow("Radius (arcsec):", radius_edit)
        result_label = QLabel("")
        form.addRow(result_label)
        fetch_btn = QPushButton("Fetch")
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)

        def do_fetch() -> None:
            try:
                ra_f = float(ra_edit.toPlainText().strip() or "0")
                dec_f = float(dec_edit.toPlainText().strip() or "0")
                radius_f = float(radius_edit.toPlainText().strip() or "5")
            except ValueError:
                result_label.setText("Invalid numbers.")
                return
            try:
                from malca.review.fetch import fetch_cone_search
                df = fetch_cone_search(ra_f, dec_f, radius_arcsec=radius_f)
                result_label.setText(f"Found {len(df)} source(s).")
            except Exception as e:
                result_label.setText(f"Error: {e}")

        fetch_btn.clicked.connect(do_fetch)
        btn_row = QHBoxLayout()
        btn_row.addWidget(fetch_btn)
        btn_row.addWidget(close_btn)
        form.addRow(btn_row)
        dlg.exec()
