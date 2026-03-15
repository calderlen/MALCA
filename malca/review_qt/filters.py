"""Full filter sidebar for review-qt: SIDEBAR_GROUPS + numeric bounds + sort (Dash parity)."""
from __future__ import annotations

import math
from pathlib import Path

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QComboBox,
        QGroupBox, QScrollArea, QPushButton, QDoubleSpinBox, QListWidget,
        QListWidgetItem, QAbstractItemView, QFrame,
    )
except ImportError:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QComboBox,
        QGroupBox, QScrollArea, QPushButton, QDoubleSpinBox, QListWidget,
        QListWidgetItem, QAbstractItemView, QFrame,
    )

from malca.review.filter_schema import SIDEBAR_GROUPS
from malca.review.store import get_numeric_bounds, get_distinct_values


def _col_id(col: str) -> str:
    return col.replace("_", "-")


def _normalize_numeric_filter_value(
    fkey: str,
    raw_value: float | None,
    bounds_data: dict | None,
) -> float | None:
    """Treat full-range numeric values as unset (Dash parity)."""
    if raw_value is None:
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    try:
        if not math.isfinite(value):
            return None
    except (TypeError, ValueError):
        return None
    if fkey.startswith("min_"):
        col, bound_key = fkey[4:], "min"
    elif fkey.startswith("max_"):
        col, bound_key = fkey[4:], "max"
    else:
        return value
    info = (bounds_data or {}).get(col) or {}
    bound = info.get(bound_key)
    if bound is None:
        return None
    try:
        bound_value = float(bound)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(bound_value):
        return None
    abs_tol = 1e-9 * max(1.0, abs(bound_value))
    if math.isclose(value, bound_value, rel_tol=1e-9, abs_tol=abs_tol):
        return None
    return value


class FilterSidebarWidget(QWidget):
    """Full filter sidebar: only unreviewed, require failed_any, all SIDEBAR_GROUPS, sort_cols, sort_desc."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._numeric_bounds: dict = {}
        self._bool_combos: dict[str, QComboBox] = {}
        self._num_min: dict[str, QDoubleSpinBox] = {}
        self._num_max: dict[str, QDoubleSpinBox] = {}
        self._text_combos: dict[str, QComboBox] = {}
        self._select_lists: dict[str, QListWidget] = {}
        self._sort_list: QListWidget | None = None
        self._sort_desc: QCheckBox | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._filter_only_unreviewed = QCheckBox("Only unreviewed")
        layout.addWidget(self._filter_only_unreviewed)
        self._filter_failed_any_false = QCheckBox("Require failed_any=False")
        layout.addWidget(self._filter_failed_any_false)

        self._bounds_btn = QPushButton("Refresh Slider Bounds")
        self._bounds_status = QLabel("Sliders load on refresh.")
        self._bounds_status.setStyleSheet("font-size: 10px; color: #7d91a6;")
        layout.addWidget(self._bounds_btn)
        layout.addWidget(self._bounds_status)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(0, 0, 0, 0)

        for grp_name, items in SIDEBAR_GROUPS:
            grp = QGroupBox(grp_name)
            grp.setCheckable(False)
            grp_layout = QVBoxLayout(grp)
            for ftype, col in items:
                if ftype == "bool":
                    combo = QComboBox()
                    combo.addItems(["Any", "True", "False", "Unset"])
                    combo.setCurrentText("Any")
                    self._bool_combos[col] = combo
                    grp_layout.addWidget(QLabel(f"{col}:"))
                    grp_layout.addWidget(combo)
                elif ftype == "num":
                    row = QHBoxLayout()
                    row.addWidget(QLabel(f"{col}:"))
                    spin_min = QDoubleSpinBox()
                    spin_min.setRange(-1e30, 1e30)
                    spin_min.setSpecialValueText("—")
                    spin_min.setValue(-1e30)
                    spin_min.setDecimals(6)
                    spin_min.setMinimumWidth(70)
                    spin_max = QDoubleSpinBox()
                    spin_max.setRange(-1e30, 1e30)
                    spin_max.setSpecialValueText("—")
                    spin_max.setValue(1e30)
                    spin_max.setDecimals(6)
                    spin_max.setMinimumWidth(70)
                    self._num_min[col] = spin_min
                    self._num_max[col] = spin_max
                    row.addWidget(spin_min)
                    row.addWidget(spin_max)
                    grp_layout.addLayout(row)
                elif ftype == "text":
                    combo = QComboBox()
                    combo.setEditable(True)
                    combo.addItem("Any")
                    combo.setCurrentText("Any")
                    self._text_combos[col] = combo
                    grp_layout.addWidget(QLabel(f"{col}:"))
                    grp_layout.addWidget(combo)
                elif ftype == "select":
                    lst = QListWidget()
                    lst.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
                    lst.setMaximumHeight(80)
                    self._select_lists[col] = lst
                    grp_layout.addWidget(QLabel(f"{col} (exclude):"))
                    grp_layout.addWidget(lst)
            inner_layout.addWidget(grp)
        inner_layout.addStretch()
        scroll.setWidget(inner)
        layout.addWidget(scroll)

        layout.addWidget(QLabel("Sort by (multi):"))
        self._sort_list = QListWidget()
        self._sort_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        sort_options = ["candidate_id", "interest_score", "updated_at", "review_pass"]
        for _, items in SIDEBAR_GROUPS:
            for ftype, col in items:
                if ftype == "num" and col not in sort_options:
                    sort_options.append(col)
        for opt in sort_options:
            self._sort_list.addItem(opt)
        for i in range(self._sort_list.count()):
            if self._sort_list.item(i).text() == "candidate_id":
                self._sort_list.item(i).setSelected(True)
                break
        layout.addWidget(self._sort_list)
        self._sort_desc = QCheckBox("Descending")
        layout.addWidget(self._sort_desc)
        self._refresh_btn = QPushButton("Refresh Queue")
        layout.addWidget(self._refresh_btn)

    def set_numeric_bounds(self, bounds: dict) -> None:
        self._numeric_bounds = bounds
        for col, spin_min in self._num_min.items():
            info = bounds.get(col) or {}
            lo = info.get("min")
            hi = info.get("max")
            if lo is not None and hi is not None:
                spin_min.setRange(lo, hi)
                spin_min.setValue(lo)
                spin_min.setSpecialValueText("—")
            if col in self._num_max:
                spin_max = self._num_max[col]
                if lo is not None and hi is not None:
                    spin_max.setRange(lo, hi)
                    spin_max.setValue(hi)
                    spin_max.setSpecialValueText("—")

    def load_numeric_bounds_sync(self, conn, columns: list[str] | None = None, **kwargs) -> dict:
        if columns is None:
            columns = list(self._num_min.keys())
        bounds = get_numeric_bounds(conn, columns=columns, **kwargs)
        self.set_numeric_bounds(bounds)
        return bounds

    def get_filter_params(self) -> dict:
        out: dict = {}
        if self._filter_only_unreviewed.isChecked():
            out["only_unreviewed"] = True
        if self._filter_failed_any_false.isChecked():
            out["require_failed_any_false"] = True
        mode_map = {"Any": None, "True": 1, "False": 0, "Unset": "unset"}
        for col, combo in self._bool_combos.items():
            mode = combo.currentText()
            val = mode_map.get(mode)
            key = f"{col}_mode"
            if val == "unset":
                out[key] = "Unset"
            elif val is not None:
                out[key] = mode
        for col in self._num_min:
            spin_min = self._num_min[col]
            spin_max = self._num_max.get(col)
            vmin = spin_min.value()
            vmax = spin_max.value() if spin_max else None
            if vmin <= -1e29:
                vmin = None
            if vmax is not None and vmax >= 1e29:
                vmax = None
            vmin = _normalize_numeric_filter_value(f"min_{col}", vmin, self._numeric_bounds)
            vmax = _normalize_numeric_filter_value(f"max_{col}", vmax, self._numeric_bounds)
            if vmin is not None:
                out[f"min_{col}"] = vmin
            if vmax is not None:
                out[f"max_{col}"] = vmax
        for col, combo in self._text_combos.items():
            val = combo.currentText().strip()
            if val and val != "Any":
                out[col] = val
        for col, lst in self._select_lists.items():
            selected = [lst.item(i).text() for i in range(lst.count()) if lst.item(i).isSelected()]
            if selected:
                out[f"exclude_{col}"] = selected
        if self._sort_list:
            sort_cols = [self._sort_list.item(i).text() for i in range(self._sort_list.count()) if self._sort_list.item(i).isSelected()]
            if sort_cols:
                out["sort_cols"] = sort_cols
            else:
                out["sort_cols"] = ["candidate_id"]
        if self._sort_desc and self._sort_desc.isChecked():
            out["sort_desc"] = True
        return out

    def set_filter_params(self, params: dict) -> None:
        if params.get("only_unreviewed"):
            self._filter_only_unreviewed.setChecked(True)
        if params.get("require_failed_any_false"):
            self._filter_failed_any_false.setChecked(True)
        mode_map = {"Any": None, "True": 1, "False": 0, "Unset": "unset"}
        for col, combo in self._bool_combos.items():
            key = f"{col}_mode"
            val = params.get(key, "Any")
            if isinstance(val, bool):
                val = "True" if val else "False"
            if val in ("Any", "True", "False", "Unset"):
                combo.setCurrentText(val)
        for col in list(self._num_min.keys()):
            vmin = params.get(f"min_{col}")
            vmax = params.get(f"max_{col}")
            if vmin is not None and col in self._num_min:
                self._num_min[col].setValue(float(vmin))
            if vmax is not None and col in self._num_max:
                self._num_max[col].setValue(float(vmax))
        for col, combo in self._text_combos.items():
            val = params.get(col, "Any")
            if val and val != "Any":
                idx = combo.findText(val)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
                else:
                    combo.setCurrentText(val)
        for col, lst in self._select_lists.items():
            exc = params.get(f"exclude_{col}") or []
            for i in range(lst.count()):
                lst.item(i).setSelected(lst.item(i).text() in exc)
        sort_cols = params.get("sort_cols") or ["candidate_id"]
        if self._sort_list:
            for i in range(self._sort_list.count()):
                self._sort_list.item(i).setSelected(self._sort_list.item(i).text() in sort_cols)
        if self._sort_desc:
            self._sort_desc.setChecked(bool(params.get("sort_desc")))

    def set_bounds_status(self, text: str) -> None:
        self._bounds_status.setText(text)

    def populate_options(self, conn, **scope_kw) -> None:
        """Load distinct values for text/select filters (call when conn available)."""
        for col, combo in self._text_combos.items():
            try:
                opts = get_distinct_values(conn, col, **scope_kw)
            except Exception:
                continue
            current = combo.currentText()
            combo.clear()
            combo.addItem("Any")
            for v in opts:
                combo.addItem(str(v))
            idx = combo.findText(current) if current else 0
            combo.setCurrentIndex(max(0, idx))
        for col, lst in self._select_lists.items():
            try:
                opts = get_distinct_values(conn, col, **scope_kw)
            except Exception:
                continue
            selected = [lst.item(i).text() for i in range(lst.count()) if lst.item(i).isSelected()]
            lst.clear()
            for v in opts:
                item = QListWidgetItem(str(v))
                item.setSelected(str(v) in selected)
                lst.addItem(item)

    def refresh_btn(self) -> QPushButton:
        return self._refresh_btn

    def bounds_btn(self) -> QPushButton:
        return self._bounds_btn
